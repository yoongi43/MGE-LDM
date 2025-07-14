from torch.utils.data import Dataset
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util as at_util
from sklearn.model_selection import train_test_split

import os; opj=os.path.join
from glob import glob
from tqdm import tqdm
from time import time
import yaml
import json
import math

AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".mp4"]
MUSDB_STEMS = ['drums', 'bass', 'other', 'vocals']  # MUSDB18 stems
NUM_TRIES = 15  # Number of tries to find a valid excerpt

class ConcatDataset(Dataset):
    def __init__(self, datasets: list, shuffle=False):
        self.datasets = datasets
        self.shuffle = shuffle
        print(f"ConcatDataset: {len(self.datasets)} datasets")
        
        dset_probs = [len(ds) for ds in self.datasets]
        self.dset_probs = [len(ds) / sum(dset_probs) for ds in self.datasets]
        
    def __len__(self):
        return sum([len(ds) for ds in self.datasets])
    
    def __getitem__(self, idx):
        # print(f"GETITEM idx: {idx}")
        if self.shuffle:
            st = time()
            state = at_util.random_state(int(time()) % (2**32-1)+idx)
            dset_idx = state.choice(len(self.datasets), p=self.dset_probs)
            dset = self.datasets[dset_idx]
            idx = state.randint(len(dset))
            item = dset[idx]
            # print(f"Concat Loading Time: {time()-st:.4f} sec")
            return item
        else:
            dset = self.datasets[idx % len(self.datasets)]
            return dset[idx % len(dset)]
            

class SourceDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __len__(self):
        raise self.num_examples

    def __getitem__(self, index):
        raise NotImplementedError

    def _adjust_signal(self, signal:AudioSignal)->AudioSignal:
        """
        Adjust signal:
        - Convert to mono if necessary
        - Resample
        - Set duration: even if we load signal with the set duration, it can be shorter than the set duration.
        """
        assert self.num_channels in [1, 2]
        if self.num_channels == 1:
            signal = signal.to_mono()
        else:
            raise NotImplementedError

        # Resample
        if signal.sample_rate != self.sr:
            signal = signal.resample(self.sr)

        # Set duration
        # if signal.duration < self.sample_duration:
        #     signal = signal.zero_pad_to(int(self.sample_duration * self.sr))
        if signal.shape[-1] < self.segment_length:
            signal = signal.zero_pad_to(self.segment_length)
        
        if signal.shape[-1] > self.segment_length:
            signal = signal[..., :self.segment_length]

        return signal
    
def collate_fn_audiotools(list_of_dicts, n_splits=None):
    return at_util.collate(list_of_dicts, n_splits=n_splits)


class SingleSourceDataset_Slakh2100(SourceDataset):
    def __init__(
        self,
        *,
        data_dir:str,
        sample_rate:int,
        segment_length:int,
        split:str="train", # "train", "valid", "test"
        num_channels:int=1,
        num_examples:int=10000,
        loudness_cutoff=-35,
        mixture_prob=0.5,
        state_id:int=None,
        num_tracks_debug:int=None,
    ):
        super().__init__()
        if split == "valid":
            split = "validation"
        self.data_dir = opj(data_dir, split)
        self.sr = sample_rate
        self.segment_length = segment_length
        self.sample_duration = math.ceil(segment_length / self.sr)
        self.split = split
        self.num_channels = num_channels
        self.num_examples = num_examples
        self.state_id = state_id
        self.state = at_util.random_state(state_id)
        self.loudness_cutoff = loudness_cutoff
        self.mixture_prob = mixture_prob
        self.tracks = os.listdir(self.data_dir) # ex) Track00001, ...
        if num_tracks_debug is not None:
            self.tracks = self.tracks[:num_tracks_debug]
        self.metadata_dict = self._preload_metadata()
        
        print(f"Slakh2100/{split}: ", len(self.tracks), " tracks found")
        
    def _preload_metadata(self):
        metadata_dict = {}
        
        for ii, track in enumerate(tqdm(self.tracks, desc="Loading metadata of Slakh2100 tracks")):
            track_dir = opj(self.data_dir, track)
            with open(opj(track_dir, "metadata.yaml"), "r") as f:
                metadata_dict[track] = yaml.safe_load(f)
        return metadata_dict
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        st = time()
        state = self.state if self.state else at_util.random_state((int(time()) % (2**16)) + idx)
        track = state.choice(self.tracks)
        metadata_dict = self.metadata_dict[track]
        meta_stems = metadata_dict["stems"] # S00, S01, ...
        stem_ids = [s for s in meta_stems.keys() if meta_stems[s]['audio_rendered']==True]
        stem_ids.sort()
        
        track_dir = opj(self.data_dir, track)
        
        ## We load mixture or summation of subset of  stems randomly
        output_dict = {}
        if state.rand() < self.mixture_prob:
            mix_signal = AudioSignal.salient_excerpt(
                audio_path=opj(track_dir, "mix.wav"),
                duration=self.sample_duration,
                num_tries=NUM_TRIES,
                state=state,
                loudness_cutoff=self.loudness_cutoff
            )
            mix_signal = self._adjust_signal(mix_signal)
            mix_wav = mix_signal.audio_data
        else:
            ## Randomly mix stems
            n_stems = len(stem_ids)
            n_mix = state.randint(1, n_stems+1)
            source_idx_arr = state.permutation(n_stems)[:n_mix]
            offset = None
            # mix_signal = AudioSignal.zeros(self.sample_duration, self.sr, self.num_channels)
            # mix_signal = AudioSignal.zeros(self.segment_length)
            # mix_signal = self._adjust_signal(mix_signal)
            sig_list = []
            for ii, src_idx in enumerate(source_idx_arr):
                if ii == 0:
                    src_id = stem_ids[src_idx]
                    signal = AudioSignal.salient_excerpt(
                        audio_path=opj(track_dir, "stems", f"{src_id}.wav"),
                        duration=self.sample_duration,
                        num_tries=NUM_TRIES,
                        state=state,
                        loudness_cutoff=self.loudness_cutoff
                    )
                    offset = signal.metadata["offset"]
                    signal = self._adjust_signal(signal)
                    wav = signal.audio_data
                else:
                    src_id = stem_ids[src_idx]
                    try:
                        signal = AudioSignal(
                            opj(track_dir, "stems", f"{src_id}.wav"),
                            sample_rate=self.sr,
                            offset=offset,
                            duration=self.sample_duration
                        )
                        signal = self._adjust_signal(signal)
                        wav = signal.audio_data
                    except Exception as e:
                        print("Error: ", e) ## 
                        ## RuntimeError:
                        #   ex) Audio file /dataset/moisesdb/moisesdb_v0.1/11845abc-8ca3-4fb2-bd84-521aeeff56f4/vocals/c9c54570-bea1-47f7-9f87-b7fac40d4de5.wav with offset 219.1193999444337 and duration 3 is empty!
                        ## Load the empty signal
                        wav = torch.zeros(1, self.num_channels, self.segment_length)
                sig_list.append(wav)
                        
            mix_wav = sum(sig_list)
        output_dict['mix_wav'] = mix_wav[0]
        # print(f"Slakh2100 loading time: {time()-st:.4f} sec")
        return output_dict
    

class SingleSourceDataset_MUSDB18(SourceDataset):
    def __init__(
        self,
        *,
        data_dir, 
        sample_rate,  ## SHARED
        segment_length, ## SHARED
        split="train",  ## SHARED
        num_channels=1, ## SHARED
        num_examples=100,
        loudness_cutoff=-35,
        mixture_prob = 0.5,
        state_id=None
    ):
        super().__init__()
        """
        outputs random single mixtures of stems from the MUSDB18 dataset
        used for training the AutoEncoder model
        """
        self.data_dir = opj(data_dir, split)
        self.sr = sample_rate
        self.segment_length = segment_length
        self.sample_duration = math.ceil(segment_length / self.sr) 
        self.split = split
        self.num_channels = num_channels
        self.num_examples = num_examples ## Number of samples per epoch
        self.state_id = state_id
        self.state = at_util.random_state(state_id) if state_id else None
        self.loudness_cutoff = loudness_cutoff
        self.mixture_prob = mixture_prob

        self.track_list = os.listdir(self.data_dir) ## Song names
        self.stems = MUSDB_STEMS

        assert len(self.track_list) > 0, "No tracks found in the data directory"
        print(f"MUSDB18/{split}: ", len(self.track_list), " tracks found")

    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        st = time()
        state = self.state if self.state else at_util.random_state(int(time()) % (2**32)+idx)
        # state = self.state if self.state else np.random.RandomState()
        track = state.choice(self.track_list)
        track_dir = opj(self.data_dir, track)

        output_dict = {}
        if state.rand() < self.mixture_prob:
            mix_signal = AudioSignal.salient_excerpt(
                audio_path=opj(track_dir, 'mixture.wav'),
                duration=self.sample_duration,
                num_tries=NUM_TRIES,
                state=state,
                loudness_cutoff=self.loudness_cutoff
            )
            mix_signal = self._adjust_signal(mix_signal)
            mix_wav = mix_signal.audio_data
        else:
            # Randomly mix stems
            n_stems = len(self.stems) ## Always 4
            n_mix = state.randint(1, n_stems+1)
            source_idx_arr = state.permutation(n_stems)[:n_mix]
            offset = None
            # mix_signal = AudioSignal.zeros(self.sample_duration, self.sr, self.num_channels)
            # mix_signal = self._adjust_signal(mix_signal)
            sig_list = []

            for ii, src_idx in enumerate(source_idx_arr):
                if ii == 0:
                    src_name = self.stems[src_idx]
                    signal = AudioSignal.salient_excerpt(
                        audio_path=opj(track_dir, f"{src_name}.wav"),
                        duration=self.sample_duration,
                        num_tries=20,
                        state=state,
                        loudness_cutoff=self.loudness_cutoff
                    )
                    signal = self._adjust_signal(signal)
                    offset = signal.metadata["offset"]
                    wav = signal.audio_data
                else:
                    assert offset is not None
                    src_name = self.stems[src_idx]
                    try:
                        signal = AudioSignal(
                            opj(track_dir, f"{src_name}.wav"),
                            sample_rate=self.sr,
                            offset=offset,
                            duration=self.sample_duration
                        )
                        signal = self._adjust_signal(signal)
                        wav = signal.audio_data
                    except Exception as e:
                        print("Error: ", e)
                        wav = torch.zeros(1, self.num_channels, self.segment_length)
                sig_list.append(wav)
            mix_wav = sum(sig_list)
        output_dict['mix_wav'] = mix_wav[0]
        # print(f"MUSDB18 loading time: {time()-st:.4f} sec")
        return output_dict
    

class SingleSourceDataset_MoisesDB(SourceDataset):
    def __init__(
        self,
        *,
        data_dir,
        sample_rate,  ## SHARED
        segment_length, ## SHARED
        split="train",  ## SHARED
        num_channels=1, ## SHARED
        num_examples=100,
        loudness_cutoff=-35,
        mixture_prob = 0.5,
        state_id=None,
        ## MoisesDB specific
        split_seed=42, ## train/test split seed
        # exclude_bleed_source=True, ## Exclude bleed source when selecting source
        
    ):
        super().__init__()
        """
        outputs random single mixtures of stems from the MoisesDB dataset
        used for training the AutoEncoder model
        """
        # import pdb; pdb.set_trace()
        track_list = glob(opj(data_dir, '*'))
        track_list = sorted(track_list) 
        train_list, test_list = train_test_split(track_list, test_size=0.1, shuffle=True, random_state=split_seed)
        self.tracks = train_list if split=='train' else test_list

        self.data_dir = data_dir
        self.sr = sample_rate
        self.segment_length = segment_length
        self.sample_duration = math.ceil(segment_length / self.sr)
        self.split = split
        self.num_channels = num_channels
        self.num_examples = num_examples
        self.state_id = state_id
        self.state = at_util.random_state(state_id) if state_id else None
        self.loudness_cutoff = loudness_cutoff
        self.mixture_prob = mixture_prob

        self.metadata_dict = self._preload_metadata()

        assert len(self.tracks) > 0, "No tracks found in the data directory"
        print(f"MoisesDB/{split}: ", len(self.tracks), " tracks found")
        

    def _preload_metadata(self):
        """
        Load metadata for all tracks
        """
        metadata_dict = {}
        for track in self.tracks:
            track_dir = opj(self.data_dir, track)
            with open(opj(track_dir, "data.json"), "r") as f:
                metadata_dict[track] = json.load(f)
        return metadata_dict

    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        ## There is no mixture file in the data
        st = time()
        output_dict = {}

        state = self.state if self.state else at_util.random_state(int(time()) % (2**16) + idx)
        track = state.choice(self.tracks)
        track_dir = opj(self.data_dir, track)
        stems_list = at_util.read_sources([track_dir], ext=[".wav"])[0]
        stems_list = [dct['path'] for dct in stems_list]
        
        ## Randomly mix stems
        n_stems = len(stems_list)
        n_mix = n_stems if state.rand() < self.mixture_prob else state.randint(1, n_stems+1)
        source_idx_arr = state.permutation(n_stems)[:n_mix]
        
        sig_list = []
        offset = None
        for ii, src_idx in enumerate(source_idx_arr):
            if ii == 0:
                signal = AudioSignal.salient_excerpt(
                    audio_path=stems_list[src_idx],
                    duration=self.sample_duration,
                    num_tries=NUM_TRIES,
                    state=state,
                    loudness_cutoff=self.loudness_cutoff
                )
                offset = signal.metadata["offset"]
                signal = self._adjust_signal(signal)
                wav = signal.audio_data
            else:
                assert offset is not None
                try:
                    signal = AudioSignal(
                        stems_list[src_idx],
                        sample_rate=self.sr,
                        offset=offset,
                        duration=self.sample_duration
                    )
                    signal = self._adjust_signal(signal)
                    wav = signal.audio_data
                except Exception as e:
                    print("Error: ", e) ## 
                    ## RuntimeError: Audio file /data2/yoongi/dataset/moisesdb/moisesdb_v0.1/11845abc-8ca3-4fb2-bd84-521aeeff56f4/vocals/c9c54570-bea1-47f7-9f87-b7fac40d4de5.wav with offset 219.1193999444337 and duration 3 is empty!
                    ## Load the empty signal
                    # signal = AudioSignal.zeros(self.sample_duration, self.sr, self.num_channels)
                    wav = torch.zeros(1, self.num_channels, self.segment_length)
            sig_list.append(wav)
        mix_wav = sum(sig_list)
        output_dict['mix_wav'] = mix_wav[0]
        # print(f"MoisesDB loading time: {time()-st:.4f} sec")
        return output_dict
    
    
class SingleSourceDataset_MedleyDB2(SourceDataset):
    def __init__(
        self,
        *,
        data_dir,  ## /data2/yoongi/dataset/medleydb_v2
        sample_rate,  ## SHARED
        segment_length, ## SHARED
        split="train",  ## SHARED
        num_channels=1, ## SHARED
        num_examples=100000,
        loudness_cutoff=-35,
        mixture_prob = 0.5,
        state_id=None,
    ):
        super().__init__()
        """
        outputs random single mixtures of stems from the MedleyDB dataset
        used for training the AutoEncoder model
        """
        assert split=="train"
        
        self.data_dir = data_dir
        self.sr = sample_rate
        self.segment_length = segment_length
        self.sample_duration = math.ceil(segment_length / self.sr)
        self.split = split
        self.num_channels = num_channels
        self.num_examples = num_examples ## Number of samples per epoch
        self.state_id = state_id
        self.state = at_util.random_state(state_id) if state_id else None
        self.loudness_cutoff = loudness_cutoff
        self.mixture_prob = mixture_prob
        self.tracks = os.listdir(self.data_dir)  # ex) Allegria_Mendel...
        
        print(f"MedleyDB2/{split}: ", len(self.tracks), " tracks found")
        
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        st = time()
        output_dict = {}
        
        state = self.state if self.state else at_util.random_state((int(time()) % (2**16)) + idx)
        track = state.choice(self.tracks)
        track_dir = opj(self.data_dir, track, f"{track}_STEMS")
        stems_list = glob(opj(track_dir, "*.wav"))
        assert len(stems_list) > 0, f"No stems found in the track directory: {track_dir}"
        
        ## Randomly mix stems
        n_stems = len(stems_list)
        n_mix = n_stems if state.rand() < self.mixture_prob else state.randint(1, n_stems+1)
        source_idx_arr = state.permutation(n_stems)[:n_mix]
        
        sig_list = []
        offset = None
        for ii, src_idx in enumerate(source_idx_arr):
            if ii == 0:
                signal = AudioSignal.salient_excerpt(
                    audio_path=stems_list[src_idx],
                    duration=self.sample_duration,
                    num_tries=NUM_TRIES,
                    state=state,
                    loudness_cutoff=self.loudness_cutoff
                )
                offset = signal.metadata["offset"]
                signal = self._adjust_signal(signal)
                wav = signal.audio_data
            else:
                assert offset is not None
                try:
                    signal = AudioSignal(
                        stems_list[src_idx],
                        sample_rate=self.sr,
                        offset=offset,
                        duration=self.sample_duration
                    )
                    signal = self._adjust_signal(signal)
                    wav = signal.audio_data
                except Exception as e:
                    print("Error: ", e)
                    wav = torch.zeros(1, self.num_channels, self.segment_length)
            sig_list.append(wav)
        mix_wav = sum(sig_list)
        output_dict['mix_wav'] = mix_wav[0]
        # print(f"MedleyDB2 loading time: {time()-st:.4f} sec")
        return output_dict
    
class SingleSourceDataset_MTGJamendo(SourceDataset):
    def __init__(
        self,
        *,
        data_dir, ## /data2/yoongi/dataset/mtg_jamendo
        sample_rate,  ## SHARED
        segment_length, ## SHARED
        split="train",  ## SHARED
        num_channels=1, ## SHARED
        num_examples=100,
        loudness_cutoff=-35,
        mixture_prob = 0.5,
        state_id=None,
    ):
        super().__init__()
        """
        outputs random single mixtures of stems from the MoisesDB dataset
        used for training the AutoEncoder model
        """
        
        self.data_dir = data_dir
        self.sr = sample_rate
        self.segment_length = segment_length
        self.sample_duration = math.ceil(segment_length / self.sr)
        self.split = split
        self.num_channels = num_channels
        
        self.num_examples = num_examples ## Number of samples per epoch
        self.state_id = state_id
        self.state = at_util.random_state(state_id) if state_id else None
        self.loudness_cutoff = loudness_cutoff
        
        if split=='train':
            self.track_idx_list = [f"{ii:02d}" for ii in range(0, 98)]
        # elif split in ["valid", "validation", "test"]:
        elif split in ['validation', 'valid']:
            self.track_idx_list = [f"{ii:02d}" for ii in range(98, 100)]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        assert len(os.listdir(self.data_dir))==100
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        st = time()
        output_dict = {}
        state = self.state if self.state else at_util.random_state(int(time()) % (2**16) + idx)
        track_idx = state.choice(self.track_idx_list)
        trackname_list = os.listdir(opj(self.data_dir, track_idx))
        assert len(trackname_list) > 0, f"No tracks found in the data directory: {self.data_dir}/{track_idx}"
        trackname = state.choice(trackname_list)
        
        mix_signal = AudioSignal.salient_excerpt(
            audio_path=opj(self.data_dir, track_idx, trackname),
            duration=self.sample_duration,
            num_tries=NUM_TRIES,
            state=state,
            loudness_cutoff=self.loudness_cutoff
        )

        mix_signal = self._adjust_signal(mix_signal)
        output_dict['mix_wav'] = mix_signal.audio_data[0]
        # print(f"MTGJamendo/{self.split} loading time: {time()-st:.4f} sec")
        return output_dict
    

class SingleSourceDataset_Others(SourceDataset):
    def __init__(
        self,
        *,
        data_dir,  ## /data2/yoongi/dataset/other_tracks
        sample_rate,  ## SHARED
        segment_length, ## SHARED
        split="train",  ## SHARED
        num_channels=1, ## SHARED
        num_examples=100000,
        loudness_cutoff=-35,
        mixture_prob = 0.5,
        state_id=None,
    ):
        super().__init__()
        """
        outputs random single mixtures of stems from the MedleyDB dataset
        used for training the AutoEncoder model
        """
        assert split=="train"
        
        self.data_dir = data_dir
        self.sr = sample_rate
        self.segment_length = segment_length
        self.sample_duration = math.ceil(segment_length / self.sr)
        self.split = split
        self.num_channels = num_channels
        self.num_examples = num_examples ## Number of samples per epoch
        self.state_id = state_id
        self.state = at_util.random_state(state_id) if state_id else None
        self.loudness_cutoff = loudness_cutoff
        self.mixture_prob = mixture_prob
        self.tracks = os.listdir(self.data_dir)  # ex) Allegria_Mendel...
        
        print(f"Other tracks/{split}: ", len(self.tracks), " tracks found")
        
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        st = time()
        output_dict = {}
        
        state = self.state if self.state else at_util.random_state((int(time()) % (2**16)) + idx)
        track = state.choice(self.tracks)
        track_dir = opj(self.data_dir, track)
        stems_list = glob(opj(track_dir, "*.wav"))
        stems_list = [s for s in stems_list if "mix.wav" not in s]  # Exclude mix.wav if exists
        assert len(stems_list) > 0, f"No stems found in the track directory: {track_dir}"
        
        ## Randomly mix stems
        if state.rand() < self.mixture_prob:
            mix_signal = AudioSignal.salient_excerpt(
                audio_path=opj(track_dir, 'mix.wav'),
                duration=self.sample_duration,
                num_tries=NUM_TRIES,
                state=state,
                loudness_cutoff=self.loudness_cutoff
            )
            mix_signal = self._adjust_signal(mix_signal)
            mix_wav = mix_signal.audio_data
        
        else: ## Randomly mix stems    
            n_stems = len(stems_list)
            n_mix = state.randint(1, n_stems+1)
            source_idx_arr = state.permutation(n_stems)[:n_mix]
            
            sig_list = []
            offset = None
            for ii, src_idx in enumerate(source_idx_arr):
                if ii == 0:
                    signal = AudioSignal.salient_excerpt(
                        audio_path=stems_list[src_idx],
                        duration=self.sample_duration,
                        num_tries=NUM_TRIES,
                        state=state,
                        loudness_cutoff=self.loudness_cutoff
                    )
                    offset = signal.metadata["offset"]
                    signal = self._adjust_signal(signal)
                    wav = signal.audio_data
                else:
                    assert offset is not None
                    try:
                        signal = AudioSignal(
                            stems_list[src_idx],
                            sample_rate=self.sr,
                            offset=offset,
                            duration=self.sample_duration
                        )
                        signal = self._adjust_signal(signal)
                        wav = signal.audio_data
                    except Exception as e:
                        print("Error: ", e)
                        wav = torch.zeros(1, self.num_channels, self.segment_length)
                sig_list.append(wav)
            mix_wav = sum(sig_list)
        output_dict['mix_wav'] = mix_wav[0]
        # print(f"MedleyDB2 loading time: {time()-st:.4f} sec")
        return output_dict
    
class SingleSourceDataset_MTGJamendo(SourceDataset):
    def __init__(
        self,
        *,
        data_dir, ## /data2/yoongi/dataset/mtg_jamendo
        sample_rate,  ## SHARED
        segment_length, ## SHARED
        split="train",  ## SHARED
        num_channels=1, ## SHARED
        num_examples=100,
        loudness_cutoff=-35,
        mixture_prob = 0.5,
        state_id=None,
    ):
        super().__init__()
        """
        outputs random single mixtures of stems from the MoisesDB dataset
        used for training the AutoEncoder model
        """
        
        self.data_dir = data_dir
        self.sr = sample_rate
        self.segment_length = segment_length
        self.sample_duration = math.ceil(segment_length / self.sr)
        self.split = split
        self.num_channels = num_channels
        
        self.num_examples = num_examples ## Number of samples per epoch
        self.state_id = state_id
        self.state = at_util.random_state(state_id) if state_id else None
        self.loudness_cutoff = loudness_cutoff
        
        if split=='train':
            self.track_idx_list = [f"{ii:02d}" for ii in range(0, 98)]
        # elif split in ["valid", "validation", "test"]:
        elif split in ['validation', 'valid']:
            self.track_idx_list = [f"{ii:02d}" for ii in range(98, 100)]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        assert len(os.listdir(self.data_dir))==100
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        st = time()
        output_dict = {}
        state = self.state if self.state else at_util.random_state(int(time()) % (2**16) + idx)
        track_idx = state.choice(self.track_idx_list)
        trackname_list = os.listdir(opj(self.data_dir, track_idx))
        assert len(trackname_list) > 0, f"No tracks found in the data directory: {self.data_dir}/{track_idx}"
        trackname = state.choice(trackname_list)
        
        mix_signal = AudioSignal.salient_excerpt(
            audio_path=opj(self.data_dir, track_idx, trackname),
            duration=self.sample_duration,
            num_tries=NUM_TRIES,
            state=state,
            loudness_cutoff=self.loudness_cutoff
        )

        mix_signal = self._adjust_signal(mix_signal)
        output_dict['mix_wav'] = mix_signal.audio_data[0]
        # print(f"MTGJamendo/{self.split} loading time: {time()-st:.4f} sec")
        return output_dict

        
def create_single_dataloader_from_config(
    dataset_configs,
    split:str,
    batch_size: int,
    sample_rate: int,
    segment_length: int,
    audio_channels: int,
    ## loader params
    num_workers: int = 4,
    shuffle: bool = True,
):
    DATASET_REGISTRY = {
        'slakh2100': SingleSourceDataset_Slakh2100,
        'musdb18': SingleSourceDataset_MUSDB18,
        'moisesdb': SingleSourceDataset_MoisesDB,
        'medleydb2': SingleSourceDataset_MedleyDB2,
        'mtg_jamendo': SingleSourceDataset_MTGJamendo,
        'others': SingleSourceDataset_Others,
    }
    
    dataset_list = []
    dataset_configs = dataset_configs[split]
    for dataset_name, dataset_cfg in dataset_configs.items():
        dataset_cls = DATASET_REGISTRY[dataset_name]
        dataset = dataset_cls(
            **dataset_cfg,
            split=split,
            sample_rate=sample_rate,
            segment_length=segment_length,
            num_channels=audio_channels,
        )
        dataset_list.append(dataset)
    
    dataset_cat = ConcatDataset(dataset_list, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(
        dataset_cat,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader