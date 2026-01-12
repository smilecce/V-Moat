import numpy as np
from typing import Dict, List, Tuple, Union

import torch, torchaudio
from torch.nn import functional as F

import sys
if sys.path[0] == '/home/vmoat/userstudy_wsbnew/certification':
    from masker import Masker
# elif sys.path[0] == '/home/vmoat/vmoat_code':
#     from .masker import Masker
else:
    from .masker import Masker

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPS = np.finfo(np.float32).eps

class Audibility:
    ''' The class to compute the audibility of a noise in
    the context of a clean audio.
    '''
    def __init__(self,
                 sample_rate: int = 16000,
                 win_length: int = 2048,
                 hop_length: int = 2048,
                 n_fft: int = 2048,
                 device = 'cuda',
                 masker = None,
                 mode = 'vmoat'
                ):
        
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.device = device
        self.mode = mode

        if masker == None:
            self.msker = Masker(
                            device = device,
                            win_length = win_length,
                            hop_length = hop_length,
                            n_fft = n_fft,
                            sample_rate = sample_rate,
                            )


    def wav_stft(self, wav_delta: torch.Tensor, 
                 original_max_psd: torch.Tensor) -> torch.Tensor:
        ''' Transform wavs into spectrograms.

        It calls Masker._psd_transfrom() to perform STFT. Make sure that the 
        length of the wavs is a multiple of the window length.

        Arguments
        ---------
        wav_delta : torch.Tensor
            A batch of wavs of size (batch, 1, n_points). Make sure that
            n_points % self.win_length == 0.
        original_max_psd : torch.Tensor
            The results output by self.compute_threshold of size (Nf,).

        Returns
        -------
        wav_stft : torch.Tensor
            The batch spectrograms of size (batch, n_fft/2+1, Nf).
        '''

        assert wav_delta.size(2) % self.win_length == 0, \
            f'''Input should be trimmed to satisfying 'n_points 
            ({wav_delta.size(2)}) % self.win_length ({self.win_length}) == 0'
            '''
        
        results = []
        for i in range(wav_delta.size(0)):
            wav_stft_i = self.msker._psd_transform(
                                        wav_delta[i, :, :],
                                        original_max_psd,
                                        mode=self.mode
                                    )
            results.append(wav_stft_i)
        return torch.vstack(results)


    def compute_threshold(self, wav_x: torch.Tensor) -> Tuple[(np.ndarray, np.ndarray)]:
        ''' Compute the masking thresholds and original_max_psd of a background audio.

        Arguments
        ---------
        wav_x : torch.Tensor
            A wav of size (1, n_points).

        Returns
        -------
        theta : np.ndarray
            Masking thresholds of size (Nf, n_fft/2+1).
        original_max_psd : np.ndarray
            original_max_psd used for normalization of size (Nf,).
        '''

        theta, original_max_psd = self.msker._compute_masking_threshold(
                                            wav_x[0, :].cpu().numpy(),
                                            self.mode
                                            )
        
        return theta, original_max_psd


    def audibility(self, wav_delta: torch.Tensor, wav_x: torch.Tensor) -> np.ndarray:
        ''' Compute the audibility of a noise in the context of a background audio.

        It first computes the thresholds and original_max_psd, computes the normalize 
        spectrograms of wav_delta, and sums up all components of audibility. It uses
        two methods to sum the results.

        Arguments
        ---------
        wav_x : torch.Tensor
            A wav of size (1, n_points).

        Returns
        -------
        [[audibility1],
         [audibility2]]
            Two values of audibilitiy are returned.
        '''
        # ::params: wav_delta:    the perturbation, Tensor: (batch, 1, n)        
        # ::params: original_max_psd  Tensor: (Nf,)
        # ::params: theta             Tensor: (Nf, n_fft/2+1)

        # Compute the thresholds and original_max_psd.
        theta, original_max_psd = self.compute_threshold(wav_x)
        original_max_psd = torch.FloatTensor(original_max_psd).unsqueeze(0).to(self.device) + EPS   # avoid zeros
        theta = torch.FloatTensor(theta).to(self.device) + EPS      # avoid zeros
        
        # Compute the normalize spectrograms of wav_delta.
        wav_delta = wav_delta.to(self.device)
        power_normalized = self.wav_stft(wav_delta, original_max_psd).transpose(1,2)

        # The traditional audibility.
        audibility_1 = torch.maximum(power_normalized - theta, \
                                    torch.zeros_like(theta).to(theta.device))\
                                    .mean(dim=(1,2))
        
        # Our audibility.
        audibility_2 = 0.75*(power_normalized / theta).mean(dim=(1,2))
        return torch.vstack([audibility_1, audibility_2]).cpu().detach().numpy()  # add .detach()



class Audibility_new:
    ''' The class to compute the audibility of a noise in
    the context of a clean audio.
    '''
    def __init__(self,
                 sample_rate: int = 16000,
                 win_length: int = 2048,
                 hop_length: int = 2048,
                 n_fft: int = 2048,
                 device = 'cuda',
                 masker = None,
                 mode = 'vmoat'
                ):
        
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.device = device
        self.mode = mode

        if masker == None:
            self.msker = Masker(
                            device = device,
                            win_length = win_length,
                            hop_length = hop_length,
                            n_fft = n_fft,
                            sample_rate = sample_rate,
                            )


    def wav_stft(self, wav_delta: torch.Tensor, 
                 original_max_psd: torch.Tensor) -> torch.Tensor:
        ''' Transform wavs into spectrograms.

        It calls Masker._psd_transfrom() to perform STFT. Make sure that the 
        length of the wavs is a multiple of the window length.

        Arguments
        ---------
        wav_delta : torch.Tensor
            A batch of wavs of size (batch, 1, n_points). Make sure that
            n_points % self.win_length == 0.
        original_max_psd : torch.Tensor
            The results output by self.compute_threshold of size (Nf,).

        Returns
        -------
        wav_stft : torch.Tensor
            The batch spectrograms of size (batch, n_fft/2+1, Nf).
        '''

        assert wav_delta.size(2) % self.win_length == 0, \
            f'''Input should be trimmed to satisfying 'n_points 
            ({wav_delta.size(2)}) % self.win_length ({self.win_length}) == 0'
            '''
        
        results = []
        for i in range(wav_delta.size(0)):
            wav_stft_i = self.msker._psd_transform(
                                        wav_delta[i, :, :],
                                        original_max_psd,
                                        mode=self.mode
                                    )
            results.append(wav_stft_i)
        return torch.vstack(results)


    def compute_threshold(self, wav_x: torch.Tensor) -> Tuple[(np.ndarray, np.ndarray)]:
        ''' Compute the masking thresholds and original_max_psd of a background audio.

        Arguments
        ---------
        wav_x : torch.Tensor
            A wav of size (1, n_points).

        Returns
        -------
        theta : np.ndarray
            Masking thresholds of size (Nf, n_fft/2+1).
        original_max_psd : np.ndarray
            original_max_psd used for normalization of size (Nf,).
        '''

        theta, original_max_psd = self.msker._compute_masking_threshold(
                                            wav_x[0, :].cpu().numpy(),
                                            self.mode
                                            )
        
        return theta, original_max_psd


    def audibility(self, wav_delta: torch.Tensor, wav_x: torch.Tensor) -> np.ndarray:
        ''' Compute the audibility of a noise in the context of a background audio.

        It first computes the thresholds and original_max_psd, computes the normalize 
        spectrograms of wav_delta, and sums up all components of audibility. It uses
        two methods to sum the results.

        Arguments
        ---------
        wav_x : torch.Tensor
            A wav of size (1, n_points).

        Returns
        -------
        [[audibility1],
         [audibility2]]
            Two values of audibilitiy are returned.
        '''
        # ::params: wav_delta:    the perturbation, Tensor: (batch, 1, n)        
        # ::params: original_max_psd  Tensor: (Nf,)
        # ::params: theta             Tensor: (Nf, n_fft/2+1)

        # Compute the thresholds and original_max_psd.
        theta, original_max_psd = self.compute_threshold(wav_x)
        original_max_psd = torch.FloatTensor(original_max_psd).unsqueeze(0).to(self.device) + EPS   # avoid zeros
        theta = torch.FloatTensor(theta).to(self.device) + EPS      # avoid zeros
        
        # Compute the normalize spectrograms of wav_delta.
        wav_delta = wav_delta.to(self.device)
        power_normalized = self.wav_stft(wav_delta, original_max_psd).transpose(1,2)
        print('pn_shape', power_normalized.shape)
        print('theta_shape', theta.shape)
        # The traditional audibility.
        audibility_1 = torch.maximum(power_normalized - theta, \
                                    torch.zeros_like(theta).to(theta.device))\
                                    .mean(dim=(1,2))
        
        # Our audibility.
        audibility_2 = (power_normalized / theta).mean(dim=(1,2))
        return torch.vstack([audibility_1, audibility_2]).cpu().detach().numpy()  # add .detach()
    
    def before_audibility(self, wav_delta: torch.Tensor, wav_x: torch.Tensor) -> np.ndarray:
        ''' Compute the audibility of a noise in the context of a background audio.

        It first computes the thresholds and original_max_psd, computes the normalize 
        spectrograms of wav_delta, and sums up all components of audibility. It uses
        two methods to sum the results.

        Arguments
        ---------
        wav_x : torch.Tensor
            A wav of size (1, n_points).

        Returns
        -------
        [[audibility1],
         [audibility2]]
            Two values of audibilitiy are returned.
        '''
        # ::params: wav_delta:    the perturbation, Tensor: (batch, 1, n)        
        # ::params: original_max_psd  Tensor: (Nf,)
        # ::params: theta             Tensor: (Nf, n_fft/2+1)

        # Compute the thresholds and original_max_psd.
        theta, original_max_psd = self.compute_threshold(wav_x)
        original_max_psd = torch.FloatTensor(original_max_psd).unsqueeze(0).to(self.device) + EPS   # avoid zeros
        theta = torch.FloatTensor(theta).to(self.device) + EPS      # avoid zeros
        
        # Compute the normalize spectrograms of wav_delta.
        wav_delta = wav_delta.to(self.device)
        power_normalized = self.wav_stft(wav_delta, original_max_psd).transpose(1,2)
        return power_normalized, theta
        # # The traditional audibility.
        # audibility_1 = torch.maximum(power_normalized - theta, \
        #                             torch.zeros_like(theta).to(theta.device))\
        #                             .mean(dim=(1,2))
        
        # # Our audibility.
        # audibility_2 = (power_normalized / theta).mean(dim=(1,2))
        # return torch.vstack([audibility_1, audibility_2]).cpu().detach().numpy()  # add .detach()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    audi = Audibility_new(
                    sample_rate = 16000,
                    win_length = 2048,
                    hop_length = 2048,
                    n_fft = 2048,
                    device = device,
                    mode='vmoat'
                    )
    
    wav_file = '/home/vmoat/shibo/Code/paper1_random_smooth/smoothing-master/code/spk1_snt2.wav'
    wav_x, sr_x = torchaudio.load(wav_file)

    print(wav_x.size())
    if wav_x.size(1)%2048 != 0:
        wav_x = wav_x[:, :-(wav_x.size(1)%2048)]

    wav_delta = torch.randn_like(wav_x)/100
    wav_delta = torch.clamp(wav_delta + wav_x, min=-1, max=1)-wav_x

    temp_a = audi.audibility(wav_delta.unsqueeze(0).to(device), \
                             wav_x)
    
    print(audi.mode, 'trim', temp_a)





# def main_adv():
#     all_audibility = []

#     audi = Audibility(
#                 sample_rate = 44100,
#                 win_length = 2048,
#                 hop_length = 2048,
#                 n_fft = 2048,
#                 device = device
#                 )
    
#     wav_file = './clean4.wav'
#     # print(wav_file)
#     wav_x, sr_x = torchaudio.load(wav_file)
#     # print('type:',wav_x.dtype)

#     for i in tqdm(np.arange(0.01, 0.101, 0.01)):
#         adv_path = f"/home/vmoat/shibo/Code/adversarial_robustness_toolbox/clean4_adv/clean4_adv_eps_{i:.2f}.wav"
#         wav_adv, sr_d = torchaudio.load(adv_path)
#         assert sr_x == sr_d
        
#         wav_delta = wav_adv - wav_x
 
#         temp_a = audi.audibility(wav_delta.unsqueeze(0).to(device), \
#                                 wav_x)
#         all_audibility.append(temp_a)
    
#     return np.hstack(all_audibility)
 
# def main_adv2(adv_dir, end=4000):
#     all_audibility = []

#     audi = Audibility(
#                 sample_rate = 16000,
#                 win_length = 2048,
#                 hop_length = 2048,
#                 n_fft = 2048,
#                 device = device
#                 )
    
#     wav_file = f'/home/vmoat/shibo/Code/adversarial_robustness_toolbox/{adv_dir}.wav'
#     # print(wav_file)
#     wav_x, sr_x = torchaudio.load(wav_file)

#     for i in tqdm(np.arange(10, end, 10)):
#         adv_path = f"/home/vmoat/shibo/Code/adversarial_robustness_toolbox/{adv_dir}_adv/adv_eps_0.01_it_{i:d}.wav"
#         wav_adv, sr_d = torchaudio.load(adv_path)
#         assert sr_x == sr_d
        
#         wav_delta = wav_adv - wav_x
#         temp_a = audi.audibility(wav_delta.unsqueeze(0).to(device), \
#                                 wav_x)
#         all_audibility.append(temp_a)
    
#     return np.hstack(all_audibility)

# def main_adv3(adv_dir, end=4000, device='cuda'):
#     all_audibility = []

#     audi = Audibility(
#                 sample_rate = 16000,
#                 win_length = 2048,
#                 hop_length = 2048,
#                 n_fft = 2048,
#                 device = device
#                 )
    
#     wav_file = f'/home/vmoat/shibo/Code/adversarial_robustness_toolbox/{adv_dir.split("_")[0]}.wav'
#     # print(wav_file)
#     wav_x, sr_x = torchaudio.load(wav_file)

#     for i in tqdm(np.arange(10, end, 10)):
#         adv_path = f"/home/vmoat/shibo/Code/adversarial_robustness_toolbox/{adv_dir}/adv_it_{i:d}_*.wav"
#         adv_path = glob.glob(adv_path)[0]
#         wav_adv, sr_d = torchaudio.load(adv_path)
#         assert sr_x == sr_d
        
#         wav_delta = wav_adv - wav_x
#         temp_a = audi.audibility(wav_delta.unsqueeze(0).to(device), \
#                                 wav_x)
#         all_audibility.append(temp_a)
    
#     return np.hstack(all_audibility)

# def main_adv4(adv_dir, end=4000, device='cuda'):
#     all_audibility = []

#     audi = Audibility(
#                 sample_rate = 16000,
#                 win_length = 2048,
#                 hop_length = 2048,
#                 n_fft = 2048,
#                 device = device
#                 )
    
#     # wav_file = f'/home/vmoat/shibo/Code/adversarial_robustness_toolbox/{adv_dir.split("_")[0]}.wav'
#     wav_file = adv_dir.split("_")[0]
#     spk_id, sub_dir, utte_id = wav_file.split('-')
#     wav_file = f'/home/vmoat/LibriSpeech/test-clean/{spk_id}/{sub_dir}/{wav_file}.flac'
#     # print(wav_file)
#     wav_x, sr_x = torchaudio.load(wav_file)

#     for i in tqdm(np.arange(10, end, 10)):
#         adv_path = f"/home/vmoat/adv_test_clean/{adv_dir}/adv_it_{i:d}_*.wav"
#         adv_path = glob.glob(adv_path)[0]
#         wav_adv, sr_d = torchaudio.load(adv_path)
#         assert sr_x == sr_d
#         wav_x_temp = wav_x[:, :wav_adv.size(1)]
        
#         wav_delta = wav_adv - wav_x_temp
#         temp_a = audi.audibility(wav_delta.unsqueeze(0).to(device), \
#                                 wav_x_temp)
#         all_audibility.append(temp_a)
    
#     return np.hstack(all_audibility)

# def main_rand(repeats = 5):
#     all_audibility = []

#     audi = Audibility(
#                 sample_rate = 16000,
#                 win_length = 2048,
#                 hop_length = 2048,
#                 n_fft = 2048,
#                 device = device
#                 )

#     wav_file = './clean2.wav'
#     wav_x, sr_x = torchaudio.load(wav_file)  
#     wav_x_resample = taF.resample(wav_x, sr_x, 16000)
#     # print(f'Length: {wav_x.size(1)}-->{wav_x_resample.size(1)}')
#     wav_x = wav_x_resample

#     # repeats = 5
#     for noise_level in tqdm(np.arange(20, 100, 4)):
#         wav_delta = torch.randn([repeats] + list(wav_x.size()))/noise_level
#         wav_delta = torch.clamp(wav_x + wav_delta, min=-1, max=1)\
#                         - wav_x
#         temp_a = audi.audibility(wav_delta.to(device), wav_x)
#         all_audibility.append(temp_a)
    
#     return np.hstack(all_audibility)