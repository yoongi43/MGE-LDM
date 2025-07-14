
import typing as tp
from functools import reduce
from packaging import version

from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.cuda.amp import autocast

try:
    from flash_attn import flash_attn_func, flash_attn_kvpacked_func
except ImportError as e:
    print(e)
    print('flash_attn not installed, disabling flash_attn')
    flash_attn_kvpacked_func = None
    flash_attn_func = None

try:
    import natten
except ImportError:
    natten = None

from ..utils import exists


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


# Copied and modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/attend.py under MIT License
# License can be found in LICENSES/LICENSE_XTRANSFORMERS.txt

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)


def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, \
            f"you are passing in a sequence length of {seq_len} \
                but your absolute positional embedding has a max sequence length of {self.max_seq_len}"

        if pos is None:
            pos = torch.arange(seq_len, device=device)

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min=0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device=device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos=False,
        scale_base=512,
        interpolation_factor=1.,
        base=10000,
        base_rescale_factor=1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device=device)
        return self.forward(t)

    @autocast(enabled=False)
    def forward(self, t):
        device = self.inv_freq.device

        t = t.to(torch.float32)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        if self.scale is None:
            return freqs, 1.

        # TODO: fix this bug
        power = (torch.arange(seq_len, device=device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


@autocast(enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

    return torch.cat((t, t_unrotated), dim=-1)

# norms

class GroupNorm(nn.Module):
    def __init__(self, dim, num_groups, bias=False, fix_scale=False, eps=1e-5):
        super().__init__()
        if dim % num_groups != 0:
            raise ValueError('dimension must be divisible by number of groups')
        self.num_groups = num_groups
        self.eps = eps
        
        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))
            
        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))
            
    def forward(self, x):
        """
        
        x: (Batch, seq, channel)
        """
        # return F.group_norm(x, self.num_groups, self.gamma, self.beta, self.eps)
        x = rearrange(x, 'b t c -> b c t')
        x = F.group_norm(x, self.num_groups, self.gamma, self.beta, self.eps)
        x = rearrange(x, 'b c t -> b t c')
        return x
    

class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False):
        """
        bias-less layernorm has been shown to be more stable. 
        most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):

        return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta)

# feedforward

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: tp.Callable,
        use_conv=False,
        conv_kernel_size=3,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2) if not use_conv else nn.Conv1d(dim_in,
                                                                                  dim_out * 2, conv_kernel_size, padding=(conv_kernel_size // 2))
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        no_bias=False,
        glu=True,
        use_conv=False,
        conv_kernel_size=3,
        zero_init_output=True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # Default to SwiGLU

        activation = nn.SiLU()

        dim_out = dim if dim_out is None else dim_out

        if glu:
            linear_in = GLU(dim, inner_dim, activation)
        else:
            linear_in = nn.Sequential(
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias=not no_bias) if not use_conv else nn.Conv1d(
                    dim, inner_dim, conv_kernel_size, padding=(conv_kernel_size // 2), bias=not no_bias),
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                activation
            )

        linear_out = nn.Linear(inner_dim, dim_out, bias=not no_bias) if not use_conv else nn.Conv1d(
            inner_dim, dim_out, conv_kernel_size, padding=(conv_kernel_size // 2), bias=not no_bias)

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)

        self.ff = nn.Sequential(
            linear_in,
            Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
            linear_out,
            Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
        )

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        dim_context: tp.Optional[int] = None,
        causal: bool = False,
        zero_init_output: bool = True,
        qk_norm: bool = False,
        natten_kernel_size: tp.Optional[int] = None
    ):
        super().__init__()
        assert qk_norm is False, "we do not normalize q and k"
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal

        dim_kv = dim_context if dim_context else dim
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        self.qk_norm = qk_norm

        # Using 1d neighborhood attention
        self.natten_kernel_size = natten_kernel_size

        if not natten_kernel_size:
            self.use_pt_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')
            self.use_fa_flash = torch.cuda.is_available() and flash_attn_func
            self.sdp_kwargs = dict(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True
            )

    def flash_attn(
            self,
            q,
            k,
            v,
            mask=None,
            causal=None
    ):
        batch, heads, q_len, _, k_len, device = *q.shape, k.shape[-2], q.device
        kv_heads = k.shape[1]
        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if heads != kv_heads:
            # Repeat interleave kv_heads to match q_heads
            heads_per_kv_head = heads // kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        causal = self.causal if (causal is None) else causal

        if q_len == 1 and causal:
            causal = False

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            if mask is None:
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        row_is_entirely_masked = None

        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            mask = mask & ~causal_mask

            # protect against an entire row being masked out

            row_is_entirely_masked = ~mask.any(dim=-1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False

        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                is_causal=causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out

    def forward(
        self,
        x: torch.Tensor,
        context: tp.Optional[torch.Tensor] = None,
        mask: tp.Optional[torch.Tensor] = None,
        context_mask: tp.Optional[torch.Tensor] = None,
        rotary_pos_emb=None,
        causal: tp.Optional[bool] = None
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, exists(context)

        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h=h)

            k, v = self.to_kv(kv_input).chunk(2, dim=-1)

            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=kv_h), (k, v))
        else:
            # Use fused linear projection
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # Normalize q and k for cosine sim attention
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        if exists(rotary_pos_emb) and not has_context:
            freqs, _ = rotary_pos_emb

            q_dtype = q.dtype
            k_dtype = k.dtype

            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)

            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)

            q = q.to(q_dtype)
            k = k.to(k_dtype)

        input_mask = context_mask

        if input_mask is None and not has_context:
            input_mask = mask

        # determine masking
        masks = []
        final_attn_mask = None  # The mask that will be applied to the attention matrix, taking all masks into account

        if exists(input_mask):
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask)

        # Other masks will be added here later

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        if self.natten_kernel_size:
            if natten is None:
                raise ImportError('natten not installed, please install natten to use neighborhood attention')

            dtype_in = q.dtype
            q, k, v = map(lambda t: t.to(torch.float32), (q, k, v))

            attn = natten.functional.natten1dqk(q, k, kernel_size=self.natten_kernel_size, dilation=1)

            if exists(final_attn_mask):
                attn = attn.masked_fill(final_attn_mask, -torch.finfo(attn.dtype).max)

            attn = F.softmax(attn, dim=-1, dtype=torch.float32)

            out = natten.functional.natten1dav(attn, v, kernel_size=self.natten_kernel_size, dilation=1).to(dtype_in)

        # Prioritize Flash Attention 2
        elif self.use_fa_flash:
            assert final_attn_mask is None, 'masking not yet supported for Flash Attention 2'
            # Flash Attention 2 requires FP16 inputs
            fa_dtype_in = q.dtype
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d').to(torch.float16), (q, k, v))

            out = flash_attn_func(q, k, v, causal=causal)

            out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')

        # Fall back to PyTorch implementation
        elif self.use_pt_flash:
            out = self.flash_attn(q, k, v, causal=causal, mask=final_attn_mask)

        else:
            # Fall back to custom implementation
            if h != kv_h:
                # Repeat interleave kv_heads to match q_heads
                heads_per_kv_head = h // kv_h
                k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))

            scale = 1. / (q.shape[-1] ** 0.5)

            kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'
            dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

            i, j, dtype = *dots.shape[-2:], dots.dtype

            mask_value = -torch.finfo(dots.dtype).max

            if exists(final_attn_mask):
                dots = dots.masked_fill(~final_attn_mask, mask_value)

            if causal:
                causal_mask = self.create_causal_mask(i, j, device=device)
                dots = dots.masked_fill(causal_mask, mask_value)

            attn = F.softmax(dots, dim=-1, dtype=torch.float32)
            attn = attn.type(dtype)

            out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, ' b h n d -> b n (h d)')

        # Communicate between heads

        # with autocast(enabled=False):
        #     out_dtype = out.dtype
        #     out = out.to(torch.float32)
        #     out = self.to_out(out).to(out_dtype)

        out = self.to_out(out)

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        return out


class ConformerModule(nn.Module):
    def __init__(
        self,
        dim,
        norm_kwargs={},
    ):

        super().__init__()

        self.dim = dim

        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs)  # This is a batch norm in the original but I don't like batch norm
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')

        return x


# class TransformerBlock(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             dim_heads: int = 64,
#             cross_attend: bool = False,
#             dim_context: tp.Optional[int] = None,
#             global_cond_dim: tp.Optional[int] = None,
#             causal: bool = False,
#             zero_init_branch_outputs: bool = True,
#             conformer: bool = False,
#             layer_ix: int = -1,
#             remove_norms: bool = False,
#             attn_kwargs={},
#             ff_kwargs={},
#             norm_kwargs={}
#     ):

#         super().__init__()
#         self.dim = dim
#         self.dim_heads = dim_heads
#         self.cross_attend = cross_attend
#         self.dim_context = dim_context
#         self.causal = causal
#         self.global_cond_dim = global_cond_dim
#         self.layer_ix = layer_ix

#         self.pre_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()

#         self.self_attn = Attention(
#             dim,
#             dim_heads=dim_heads,
#             causal=causal,
#             zero_init_output=zero_init_branch_outputs,
#             **attn_kwargs
#         )

#         if cross_attend:
#             self.cross_attend_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
#             self.cross_attn = Attention(
#                 dim,
#                 dim_heads=dim_heads,
#                 dim_context=dim_context,
#                 causal=causal,
#                 zero_init_output=zero_init_branch_outputs,
#                 **attn_kwargs
#             )

#         self.ff_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
#         self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)

#         self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs) if conformer else None

#         if global_cond_dim:
#             self.to_scale_shift_gate = nn.Sequential(
#                 nn.SiLU(),
#                 nn.Linear(global_cond_dim, dim * 6, bias=False)
#             )

#             nn.init.zeros_(self.to_scale_shift_gate[1].weight)
#             # nn.init.zeros_(self.to_scale_shift_gate_self[1].bias)

#     def forward(
#         self,
#         x: torch.Tensor,
#         context: tp.Optional[torch.Tensor] = None,
#         global_cond: tp.Optional[torch.Tensor] = None,
#         mask: tp.Optional[torch.Tensor] = None,
#         context_mask: tp.Optional[torch.Tensor] = None,
#         rotary_pos_emb=None
#     ):
#         if self.global_cond_dim and self.global_cond_dim > 0 and exists(global_cond):

#             scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = self.to_scale_shift_gate(global_cond).unsqueeze(1).chunk(6, dim=-1)

#             # self-attention with adaLN
#             residual = x
#             x = self.pre_norm(x)
#             x = x * (1 + scale_self) + shift_self
#             x = self.self_attn(x, mask=mask, rotary_pos_emb=rotary_pos_emb)
#             x = x * torch.sigmoid(1 - gate_self)
#             x = x + residual

#             if exists(context):
#                 x = x + self.cross_attn(self.cross_attend_norm(x), context=context, context_mask=context_mask)

#             if self.conformer:
#                 x = x + self.conformer(x)

#             # feedforward with adaLN
#             residual = x
#             x = self.ff_norm(x)
#             x = x * (1 + scale_ff) + shift_ff
#             x = self.ff(x)
#             x = x * torch.sigmoid(1 - gate_ff)
#             x = x + residual

#         else:
#             x = x + self.self_attn(self.pre_norm(x), mask=mask, rotary_pos_emb=rotary_pos_emb)

#             if exists(context):
#                 x = x + self.cross_attn(self.cross_attend_norm(x), context=context, context_mask=context_mask)

#             if self.conformer:
#                 x = x + self.conformer(x)

#             x = x + self.ff(self.ff_norm(x))

#         return x


##! Modified: Norm type can be specified
class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        dim_heads: int = 64,
        cross_attend: bool = False,
        dim_context: tp.Optional[int] = None,
        global_cond_dim: tp.Optional[int] = None,
        causal: bool = False, 
        zero_init_branch_outputs: bool = True,
        conformer: bool = False,
        layer_ix: int = -1,
        remove_norms: bool = False,
        norm_type: tp.Literal["layernorm", "groupnorm"] = None,
        global_cond_group: int = 1,
        attn_kwargs={},
        ff_kwargs={},
        norm_kwargs={},
    ):
        """
        global_cond_dim: embed_dim*3 이미 되어 있음.
        """
        super().__init__()
        # import pdb; pdb.set_trace()
        assert norm_type is not None, "Norm type must be specified."
        assert norm_type in ["layernorm", "groupnorm"], "Norm type must be layernorm or groupnorm"
        self.norm_type = norm_type

        if norm_type == "layernorm":
            norm_cls = LayerNorm
        elif norm_type == "groupnorm":
            norm_cls = GroupNorm
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")

        self.dim = dim
        self.dim_heads = dim_heads 
        self.cross_attend = cross_attend
        self.dim_context = dim_context # cross-attention context dimension
        self.causal = causal
        self.global_cond_dim = global_cond_dim # global conditioning dimension
        self.layer_ix = layer_ix # layer index

        self.pre_norm = norm_cls(dim, **norm_kwargs) if not remove_norms else nn.Identity()

        self.self_attn = Attention(
            dim,
            dim_heads=dim_heads,
            dim_context=None,
            causal=causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs
        )

        if cross_attend:
            self.cross_attend_norm = norm_cls(dim, **norm_kwargs) if not remove_norms else nn.Identity()
            self.cross_ttn = Attention(
                dim,
                dim_heads=dim_heads,
                dim_context=dim_context,
                causal=causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs
            )

        self.ff_norm = norm_cls(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)

        self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs) if conformer else None


        self.global_cond_dim = global_cond_dim
        if global_cond_dim: #! FiLM
            ## cond: (B, 3*emb)
            ## dim: 3*emb
            # condition: label Y, timestep t
            if global_cond_group == 1:
                self.to_scale_shift_gate = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(global_cond_dim, dim*6, bias=False),
                ) # (B, 3*emb) -> (B, 3*emb*6)
                nn.init.zeros_(self.to_scale_shift_gate[1].weight)
            else:                
                self.to_scale_shift_gate = nn.Sequential(
                    nn.SiLU(),
                    Rearrange('b ge -> b ge 1'),
                    nn.Conv1d(global_cond_dim, dim*6, kernel_size=1, groups=global_cond_group, bias=False), 
                    Rearrange('b (g e) 1 -> b g e', g=global_cond_group) # e: 6*emb    
                )
                nn.init.zeros_(self.to_scale_shift_gate[2].weight)

    
    def forward(
        self,
        x: torch.Tensor,
        context: tp.Optional[torch.Tensor] = None,
        global_cond: tp.Optional[torch.Tensor] = None,
        mask: tp.Optional[torch.Tensor] = None,
        context_mask: tp.Optional[torch.Tensor] = None,
        rotary_pos_emb=None
    ):
        if self.global_cond_dim and self.global_cond_dim > 0 and exists(global_cond):
            # adaptive layer normalization (FiLM) with global conditioning
            # global_cond: (batch, global_cond_dim)
            if self.global_cond_dim == 1:
                scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff =\
                    self.to_scale_shift_gate(global_cond).unsqueeze(1).chunk(6, dim=-1)
                ## scale_self: (B, 1, 3*emb)
            else:
                ## global_cond: (B, 3*cemb)
                ## to_scale: (B, 3*emb*6) -> (B, 3, emb*6)
                ## -> (B, 3, emb*6)[:, 0, :] -> (B, 1, emb*6): 첫 번 째 track의 scale, shift, gate들.
                scale_self_group, shift_self_group, gate_self_group, scale_ff_group, shift_ff_group, gate_ff_group =\
                    self.to_scale_shift_gate(global_cond).chunk(6, dim=-1) 
                ## scale_self_group: (B, 3, emb*6) -> (B, 3, emb) ## 각 group의 scale. 
                scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff =\
                    map(lambda x: rearrange(x, 'b g e -> b 1 (g e)'), 
                        (scale_self_group, shift_self_group, gate_self_group, scale_ff_group, shift_ff_group, gate_ff_group))

                
            # scale_self: (batch, 1, dim)
            # gate_self: (batch, 1, dim), "adaLN-Zero" in the DiT paper

            # self-attention with adaLN
            residual = x # (x: (batch, seq, dim))
            # print(f"Transf idx: {self.layer_ix} || x: {x.shape}") # x: (B, 65, 1536)
            # import pdb; pdb.set_trace()
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, mask=mask, rotary_pos_emb=rotary_pos_emb)
            x = x * torch.sigmoid(1 - gate_self) # "adaLN-Zero" in the DiT paper
            x = x + residual 

            if exists(context):
                x = x + self.cross_attend(self.cross_attend_norm(x), context, mask=mask, context_mask=context_mask)
            
            if self.conformer:
                x = self.conformer(x)

            # feedforward with adaLN
            residual = x # x: (batch, seq, dim)
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x)
            x = x * torch.sigmoid(1 - gate_ff)
            x = x + residual
        
        else:
            x = x + self.self_attn(self.pre_norm(x), mask=mask, rotary_pos_emb=rotary_pos_emb)

            if exists(context):
                x = x + self.cross_attend(self.cross_attend_norm(x), context, mask=mask, context_mask=context_mask)
            
            if self.conformer:
                x = self.conformer(x)

            x = x + self.ff(self.ff_norm(x))
        
        return x

class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        # *,
        dim_in: tp.Optional[int] = None,
        dim_out: tp.Optional[int] = None,
        dim_heads: int = 64,
        cross_attend: bool = False,
        cond_token_dim: tp.Optional[int] = None,
        global_cond_dim: tp.Optional[int] = None,
        norm_type: tp.Literal["layernorm", "groupnorm"] = "layernorm",
        global_cond_group: int = 1, ## Grouped Linear Layer for FiLM conditioning
        # Below are the given arguments
        causal: bool = False,
        rotary_pos_emb: bool = True, ## We have to consider this.
        zero_init_branch_outputs: bool = True,
        conformer: bool = False,
        use_sinusoidal_emb: bool = False,
        use_abs_pos_emb: bool = False,
        abs_pos_emb_max_length: int = 10000,
        norm_kwargs: dict = {},
        **kwargs
    ):
        super().__init__()
        assert not (use_sinusoidal_emb and use_abs_pos_emb), "Can't select both of sinusoidal/abs positional embedding type."

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out else nn.Identity()

        self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32)) if rotary_pos_emb else None

        self.pos_type = None
        self.pos_emb = None
        if use_sinusoidal_emb:
            self.pos_type = 'sinusoidal'
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        elif use_abs_pos_emb:
            self.pos_type = 'abs'
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length)
            
        for i in range(depth):
            self.layers.append(
                TransformerBlock(
                    dim=dim,
                    dim_heads=dim_heads,
                    cross_attend=cross_attend,
                    dim_context=cond_token_dim,
                    global_cond_dim=global_cond_dim,
                    causal=causal,
                    zero_init_branch_outputs=zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    remove_norms=False,
                    norm_type=norm_type,
                    global_cond_group=global_cond_group,
                    norm_kwargs=norm_kwargs,
                    **kwargs
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: tp.Optional[torch.Tensor] = None,
        prepend_embeds: tp.Optional[torch.Tensor] = None,
        prepend_mask: tp.Optional[torch.Tensor] = None,
        global_cond: tp.Optional[torch.Tensor] = None,
        return_info: bool = False,
        **kwargs
    ):
        # x: (batch, seq, dim)
        batch, seq, device = *x.shape[:2], x.device

        info = {"hidden_states": []}

        x = self.project_in(x)

        if exists(prepend_embeds):
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            x = torch.cat((prepend_embeds, x), dim=-2) # 

            if exists(prepend_mask) or exists(mask):
                mask = mask if exists(mask) else torch.ones((batch, seq), device=device, dtype=torch.bool)
                prepend_mask = prepend_mask if exists(prepend_mask) else torch.ones((batch, prepend_length), device=device, dtype=torch.bool)
                mask = torch.cat((prepend_mask, mask), dim=-1)

        # Attention layers

        rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1]) if self.rotary_pos_emb else None

        if self.pos_emb:
            x = x + self.pos_emb(x)

        # Iterate over the transformer layers
        for layer in self.layers:

            # x = checkpoint(layer, x, rotary_pos_emb=rotary_pos_emb, global_cond=global_cond, **kwargs)
            x = layer(x, rotary_pos_emb=rotary_pos_emb, global_cond=global_cond, **kwargs)

            if return_info:
                info["hidden_states"].append(x)

        x = self.project_out(x)

        return (x, info) if return_info else x
