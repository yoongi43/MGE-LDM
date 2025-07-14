from .autoencoders import OobleckEncoder, OobleckDecoder, AudioAutoencoder
import typing as tp

"""
AutoEncoder Configurations
"""

def create_bottleneck_from_config(bottleneck_config):
    bottleneck_type = bottleneck_config['type']

    if bottleneck_type == 'tanh':
        from .bottleneck import TanhBottleneck
        bottleneck = TanhBottleneck()
    elif bottleneck_type == 'vae':  ##! We use Oobleck's VAE bottleneck, others are not experimented with.
        from .bottleneck import VAEBottleneck
        bottleneck = VAEBottleneck()
    elif bottleneck_type == 'rvq':
        raise NotImplementedError("Discrete bottlenecks are not supported.")
        from .bottleneck import RVQBottleneck
        quantizer_params = {
            "dim": 128,
            "codebook_size": 1024,
            "num_quantizers": 8,
            "decay": 0.99,
            "kmeans_init": True,
            "kmeans_iters": 50,
            "threshold_ema_dead_code": 2,
        }
        quantizer_params.update(bottleneck_config["config"])
        bottleneck = RVQBottleneck(**quantizer_params)
    elif bottleneck_type == "dac_rvq":
        raise NotImplementedError("Discrete bottlenecks are not supported.")
        from .bottleneck import DACRVQBottleneck
        bottleneck = DACRVQBottleneck(**bottleneck_config["config"])
    elif bottleneck_type == 'rvq_vae':
        raise NotImplementedError("Discrete bottlenecks are not supported.")
        from .bottleneck import RVQVAEBottleneck
        quantizer_params = {
            "dim": 128,
            "codebook_size": 1024,
            "num_quantizers": 8,
            "decay": 0.99,
            "kmeans_init": True,
            "kmeans_iters": 50,
            "threshold_ema_dead_code": 2,
        }
        quantizer_params.update(bottleneck_config["config"])
        bottleneck = RVQVAEBottleneck(**quantizer_params)
    elif bottleneck_type == 'dac_rvq_vae':
        raise NotImplementedError("Discrete bottlenecks are not supported.")
        from .bottleneck import DACRVQVAEBottleneck
        bottleneck = DACRVQVAEBottleneck(**bottleneck_config["config"])
    elif bottleneck_type == 'l2_norm':
        from .bottleneck import L2Bottleneck
        bottleneck = L2Bottleneck()
    elif bottleneck_type == "wasserstein":
        from .bottleneck import WassersteinBottleneck
        bottleneck = WassersteinBottleneck(**bottleneck_config.get("config", {}))
    elif bottleneck_type == "fsq":
        raise NotImplementedError("Discrete bottlenecks are not supported.")
        from .bottleneck import FSQBottleneck
        bottleneck = FSQBottleneck(**bottleneck_config["config"])
    else:
        raise NotImplementedError(f'Unknown bottleneck type: {bottleneck_type}')

    requires_grad = bottleneck_config.get('requires_grad', True)
    if not requires_grad:
        for param in bottleneck.parameters():
            param.requires_grad = False

    return bottleneck


def create_encoder_from_config(encoder_config: tp.Dict[str, tp.Any]):
    encoder = OobleckEncoder(**encoder_config)
    requires_grad = encoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in encoder.parameters():
            param.requires_grad = False
    return encoder


def create_decoder_from_config(decoder_config: tp.Dict[str, tp.Any]):
    decoder = OobleckDecoder(**decoder_config)
    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False
    return decoder


def create_autoencoder_from_config(config: tp.Dict[str, tp.Any]):

    ae_config = config["model"]
    # import pdb; pdb.set_trace()
    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])
    bottleneck = ae_config.get("bottleneck", None) ## "vae"

    latent_dim = ae_config["latent_dim"]
    downsampling_ratio = ae_config["downsampling_ratio"]
    io_channels = ae_config["io_channels"]
    sample_rate = config["sample_rate"]

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)
    pretransform = ae_config.get("pretransform", None)

    if pretransform:
        raise NotImplementedError("Pretransform is not implemented.")
        # pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck:
        bottleneck = create_bottleneck_from_config(bottleneck)

    soft_clip = ae_config["decoder"].get("soft_clip", False)

    return AudioAutoencoder(
        encoder,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=pretransform,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip
    )
    
    
"""
Diffusion Configurations
"""
def create_multi_conditioner_from_config(conditioners_configs_dict: tp.Dict[str, tp.Any]):
    from .conditioners import MultiConditioner
    
    conditioners = {}
    for conditioner_key, conditioner_config in conditioners_configs_dict.items():
        cond_key = conditioner_config["cond_key"] 
        assert cond_key in ["audio_cond", "prompt_cond"]
        if conditioner_key == "clap_text":
            from .conditioners import CLAPTextConditioner
            conditioners[cond_key] = CLAPTextConditioner(**conditioner_config['config'])
        elif conditioner_key == "clap_audio":
            ## TODO: Implement on-the-fly audio conditioning
            from .conditioners import CLAPAudioConditioner
            conditioners[cond_key] = CLAPAudioConditioner(**conditioner_config['config'])
        else:
            raise NotImplementedError(f"Unknown conditioner type: {conditioner_key}")
    return MultiConditioner(conditioners)


def create_pretransform_from_config(pretransform_conig, sample_rate):
    from .pretransforms import AutoencoderPretransform
    
    autoencoder_config = {"sample_rate": sample_rate, "model": pretransform_conig['ae_config']}
    autoencoder = create_autoencoder_from_config(autoencoder_config)
    
    scale = pretransform_conig.get("scale", 1.0)
    model_half = pretransform_conig.get("model_half", False)
    iterate_batch = pretransform_conig.get("iterate_batch", False)
    chunked = pretransform_conig.get("chunked", False)
    
    pretransform = AutoencoderPretransform(
        autoencoder,
        scale=scale,
        model_half=model_half,
        iterate_batch=iterate_batch,
        chunked=chunked
    )
    
    enable_grad = pretransform_conig.get('enable_grad', False)
    pretransform.enable_grad = enable_grad
    pretransform.eval().requires_grad_(pretransform.enable_grad)
    return pretransform

def create_mgeldm_from_config(config: tp.Dict[str, tp.Any]):
    from .diffusion import MGELDM
    
    # model = MGELDM(**config)
    conditioner = create_multi_conditioner_from_config(config["conditioner"])
    pretransform = config.get("pretransform", None)
    
    ## MGE-LDM: we use pretransform only for logging wavfiles in validation step (Decoder)
    ## 
    assert pretransform is not None, "Pretransform must be specified in the config."
    
    if pretransform:
        pretransform = create_pretransform_from_config(pretransform, config["sample_rate"])
        # min_input_length = pretransform.downsampling_ratio
    else:
        # min_input_length = 1
        pass
    
    return MGELDM(
        conditioner=conditioner,
        pretransform=pretransform,
        # min_input_length=min_input_length,
        sample_rate=config["sample_rate"],
        **config["MGELDM"]
    )

