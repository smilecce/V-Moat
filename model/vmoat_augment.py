import os
import numpy as np
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.processing.speech_augmentation import (
    SpeedPerturb,
    DropFreq,
    DropChunk,
    AddBabble,
    AddNoise,
    AddReverb,
)
from speechbrain.utils.torch_audio_backend import check_torchaudio_backend

# from vmoat_code.certification.audibility import Audibility
import sys
sys.path.append('..')
from certification.vmoat import Vmoat
# import vmoat_code.certification.utils

check_torchaudio_backend()

OPENRIR_URL = "http://www.openslr.org/resources/28/rirs_noises.zip"


class VmoatCorrupt(torch.nn.Module):
    """ Psychoacoustics-oriented corruptions for speech signals.

    Arguments
    ---------
    reverb_prob : float from 0 to 1
        The probability that each batch will have reverberation applied.
    babble_prob : float from 0 to 1
        The probability that each batch will have babble added.
    noise_prob : float from 0 to 1
        The probability that each batch will have noise added.
    openrir_folder : str
        If provided, download and prepare openrir to this location. The
        reverberation csv and noise csv will come from here unless overridden
        by the ``reverb_csv`` or ``noise_csv`` arguments.
    openrir_max_noise_len : float
        The maximum length in seconds for a noise segment from openrir. Only
        takes effect if ``openrir_folder`` is used for noises. Cuts longer
        noises into segments equal to or less than this length.
    reverb_csv : str
        A prepared csv file for loading room impulse responses.
    noise_csv : str
        A prepared csv file for loading noise data.
    noise_num_workers : int
        Number of workers to use for loading noises.
    babble_speaker_count : int
        Number of speakers to use for babble. Must be less than batch size.
    babble_snr_low : int
        Lowest generated SNR of reverbed signal to babble.
    babble_snr_high : int
        Highest generated SNR of reverbed signal to babble.
    noise_snr_low : int
        Lowest generated SNR of babbled signal to noise.
    noise_snr_high : int
        Highest generated SNR of babbled signal to noise.
    rir_scale_factor : float
        It compresses or dilates the given impulse response.
        If ``0 < rir_scale_factor < 1``, the impulse response is compressed
        (less reverb), while if ``rir_scale_factor > 1`` it is dilated
        (more reverb).
    reverb_sample_rate : int
        Sample rate of input audio signals (rirs) used for reverberation.
    noise_sample_rate: int
        Sample rate of input audio signals used for adding noise.
    clean_sample_rate: int
        Sample rate of original (clean) audio signals.
    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> corrupter = VmoatCorrupt(babble_speaker_count=9)
    >>> feats = corrupter(inputs, torch.ones(10))
    """


    def __init__(
        self,
        reverb_prob=1.0,
        babble_prob=1.0,
        noise_prob=1.0,
        openrir_folder=None,
        openrir_max_noise_len=None,
        reverb_csv=None,
        noise_csv=None,
        sigma =None,    # The sigma to sample spectrogram noise
        noise_num_workers=0,
        babble_speaker_count=0,
        babble_snr_low=0,
        babble_snr_high=0,
        noise_snr_low=0,
        noise_snr_high=0,
        rir_scale_factor=1.0,
        reverb_sample_rate=16000,
        noise_sample_rate=16000,
        clean_sample_rate=16000,
    ):
        super().__init__()
        
        # The sigma to sample spectrogram noise
        self.sigma = sigma 

        # Download and prepare openrir
        if openrir_folder and (not reverb_csv or not noise_csv):

            open_reverb_csv = os.path.join(openrir_folder, "reverb.csv")
            open_noise_csv = os.path.join(openrir_folder, "noise.csv")
            _prepare_openrir(
                openrir_folder,
                open_reverb_csv,
                open_noise_csv,
                openrir_max_noise_len,
            )

            # Specify filepath and sample rate if not specified already
            if not reverb_csv:
              
              
                reverb_csv = open_reverb_csv
                reverb_sample_rate = 16000

            if not noise_csv:
                noise_csv = open_noise_csv
                noise_sample_rate = 16000

        # Initialize corrupters
        if reverb_csv is not None and reverb_prob > 0.0:
            self.add_reverb = AddReverb(
                reverb_prob=reverb_prob,
                csv_file=reverb_csv,
                rir_scale_factor=rir_scale_factor,
                reverb_sample_rate=reverb_sample_rate,
                clean_sample_rate=clean_sample_rate,
            )

        if babble_speaker_count > 0 and babble_prob > 0.0:
            self.add_babble = AddBabble(
                mix_prob=babble_prob,
                speaker_count=babble_speaker_count,
                snr_low=babble_snr_low,
                snr_high=babble_snr_high,
            )

        if noise_csv is not None and noise_prob > 0.0:
            # self.add_noise = AddNoise(
            #     mix_prob=noise_prob,
            #     csv_file=noise_csv,
            #     num_workers=noise_num_workers,
            #     snr_low=noise_snr_low,
            #     snr_high=noise_snr_high,
            #     noise_sample_rate=noise_sample_rate,
            #     clean_sample_rate=clean_sample_rate,
            # )
            print(f"vmoat_augment - Start augment samples with V-Moat, clip to [-1, 1], sigma={self.sigma}.")
            bc = blank_classifier('cpu')
            self.add_noise = Vmoat(bc, num_classes=0, sigma=self.sigma)
        

    def forward(self, waveforms, lengths, c_mk):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort.
        """
        # Augmentation
        with torch.no_grad():
            # 从上往下一个一个的判断是否要加 reverb babble noise
            if hasattr(self, "add_reverb"):     # hasattr() 函数用于判断对象是否包含对应的属性。 
                try:
                    waveforms = self.add_reverb(waveforms, lengths)
                except Exception:
                    pass
            if hasattr(self, "add_babble"):
                waveforms = self.add_babble(waveforms, lengths)
            if hasattr(self, "add_noise"):
                # waveforms = self.add_noise(waveforms, lengths)    #【before】
                waveforms = vmoat_add_noise(self.add_noise, waveforms, lengths, c_mk, self.sigma)

        return waveforms


# 自己加的函数     为vmoat的加Noise：noise的形式是根据我们得到的心理阈值来设置的
# 这个以比较优雅的写法是 写一个class 然后借鉴的是AddNoise这个class  牵扯的import比较多，感觉没太大的需要，
# 采取直接写了一个函数的方法
def vmoat_add_noise(vmoat, wav_x, length, c_mk, sigma_max):
    ''' Add Psychoacoustics-oriented noises.

    Arguments
    ---------
    vmoat : Vmoat
        A Vmoat instance for noise sampling.
    wav_x : torch.Tensor
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    lengths : tensor
        Shape should be a single dimension, `[batch]`.
    
    Returns
    -------
    x_add_noise: torch.Tensor
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    '''
    # x_add_noise = torch.zeros_like(wav_x)
    # vmoat.device = wav_x.device
    # print('vmoat_augment - wav_x', wav_x.size())

    # uniformly sample from [0, sigma_max)
    # sigmas = (torch.rand([c_mk.size(0), 1, 1]) * sigma_max).to(wav_x.device)
    
    # noise = vmoat.batch_presample_noise(c_mk/(sigmas**2), info=False)
    noise = vmoat.batch_presample_noise(c_mk, info=False)
    x_add_noise = torch.clamp(wav_x + noise.squeeze(1).to(wav_x.device), min=-1, max=1)
    # for idx in range(wav_x.size(0)):
        # c_mk = vmoat._noise_dist(wav_x[idx:idx+1, :], info=False).to(vmoat.device)
        # c_mk = c_mk.to(vmoat.device)
        # noise = vmoat._presample_noise(c_mk[idx, :, :], batchsize = 1, info=False)
        # noise = vmoat._presample_noise(c_mk, batchsize = 1, info=False)

        # if noise.isnan().any() or noise.isinf().any():
        #     noise[noise.isnan()|noise.isinf()] = 0
        #     print('vmoat_augment - WARNING: noises have NAN or INF values.')

        # x_add_noise[idx:idx+1, :] = torch.clamp(wav_x[idx:idx+1, :] + noise.to(wav_x.device), 
                                                # min=-1, max=1)
        # tmp = wav_x[idx:idx+1, :] + noise.to(wav_x.device)
        # x_add_noise[idx:idx+1, :] = tmp/tmp.abs().max()

        # torchaudio.save('Test_vmoat_augment.wav', x_add_noise[idx:idx+1, :].cpu(), 16000)
        # print('vmoat_augment - saved')
        # exit()
        # if not torch.isfinite(x_add_noise).all():
        #     print('vmoat_augment - WARNING: x_add_noise has NAN or INF values.')
        
    return x_add_noise


class blank_classifier(torch.nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device


def _prepare_openrir(folder, reverb_csv, noise_csv, max_noise_len):
    """ Prepare the openrir dataset for adding reverb and noises.

    Arguments
    ---------
    folder : str
        The location of the folder containing the dataset.
    reverb_csv : str
        Filename for storing the prepared reverb csv.
    noise_csv : str
        Filename for storing the prepared noise csv.
    max_noise_len : float
        The maximum noise length in seconds. Noises longer
        than this will be cut into pieces.
    """

    # Download and unpack if necessary
    filepath = os.path.join(folder, "rirs_noises.zip")

    if not os.path.isdir(os.path.join(folder, "RIRS_NOISES")):
        download_file(OPENRIR_URL, filepath, unpack=True)
    else:
        download_file(OPENRIR_URL, filepath)

    # Prepare reverb csv if necessary
    if not os.path.isfile(reverb_csv):
        rir_filelist = os.path.join(
            folder, "RIRS_NOISES", "real_rirs_isotropic_noises", "rir_list"
        )
        _prepare_csv(folder, rir_filelist, reverb_csv)

    # Prepare noise csv if necessary
    if not os.path.isfile(noise_csv):
        noise_filelist = os.path.join(
            folder, "RIRS_NOISES", "pointsource_noises", "noise_list"
        )
        _prepare_csv(folder, noise_filelist, noise_csv, max_noise_len)


def _prepare_csv(folder, filelist, csv_file, max_length=None):
    """ Iterate a set of wavs and write the corresponding csv file.
    
    Arguments
    ---------
    folder : str
        The folder relative to which the files in the list are listed.
    filelist : str
        The location of a file listing the files to be used.
    csvfile : str
        The location to use for writing the csv file.
    max_length : float
        The maximum length in seconds. Waveforms longer
        than this will be cut into pieces.
    """
    try:
        # make sure all processing reached here before main preocess create csv_file
        sb.utils.distributed.ddp_barrier()
        if sb.utils.distributed.if_main_process():
            with open(csv_file, "w") as w:
                w.write("ID,duration,wav,wav_format,wav_opts\n\n")
                for line in open(filelist):

                    # Read file for duration/channel info
                    filename = os.path.join(folder, line.split()[-1])
                    signal, rate = torchaudio.load(filename)

                    # Ensure only one channel
                    if signal.shape[0] > 1:
                        signal = signal[0].unsqueeze(0)
                        torchaudio.save(filename, signal, rate)

                    ID, ext = os.path.basename(filename).split(".")
                    duration = signal.shape[1] / rate

                    # Handle long waveforms
                    if max_length is not None and duration > max_length:
                        # Delete old file
                        os.remove(filename)
                        for i in range(int(duration / max_length)):
                            start = int(max_length * i * rate)
                            stop = int(
                                min(max_length * (i + 1), duration) * rate
                            )
                            new_filename = (
                                filename[: -len(f".{ext}")] + f"_{i}.{ext}"
                            )
                            torchaudio.save(
                                new_filename, signal[:, start:stop], rate
                            )
                            csv_row = (
                                f"{ID}_{i}",
                                str((stop - start) / rate),
                                "$rir_root" + new_filename[len(folder) :],
                                ext,
                                "\n",
                            )
                            w.write(",".join(csv_row))
                    else:
                        w.write(
                            ",".join(
                                (
                                    ID,
                                    str(duration),
                                    "$rir_root" + filename[len(folder) :],
                                    ext,
                                    "\n",
                                )
                            )
                        )
    finally:
        sb.utils.distributed.ddp_barrier()