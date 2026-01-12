from math import ceil
from tqdm import tqdm

import torch, torchaudio
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm, binom_test

import sys
if sys.path[0] == '/home/vmoat/vmoat_code/certification':
    from audibility import Audibility
    import utils # as utils
else:
    from .audibility import Audibility
    from . import utils # as utils

from typing import Dict, List, Tuple, Union


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
PADDING = 0
EPS = np.finfo(np.float32).eps

# Energy correction factor of hamming window.
ECF = 1.59**2

class Vmoat(object):
    ''' The class to certify models with the psychoacoustic constraint.
    '''

    ABSTAIN = -1

    def __init__(self,
                 base_classifier: torch.nn.Module, 
                 num_classes: int, 
                 sigma: float
                 ) -> None:
        self.base_classifier = base_classifier
        self.device = self.base_classifier.device

        self.num_classes = num_classes      
        self.sigma = sigma
        self.audi = Audibility(
                            sample_rate = 16000,
                            win_length = 2048,
                            hop_length = 2048,
                            n_fft = 2048,
                            device=self.device,
                            mode='vmoat'
                            )
        
    def _noise_dist(self, wav_x: torch.Tensor, info: bool=True):
        ''' Obtain the parameters (C[m,k]) of Gaussian distribution.

        Arguments
        ---------
        wav_x : torch.Tensor
            The clean audio with a size of [1, n_points].

        Returns
        -------
        c_mk : torch.Tensor
            The variance (square of std) of Gaussian distribution.
        '''
        # Compute the theta (threshold) and the original_max_psd of the wav.
        theta, original_max_psd = self.audi.compute_threshold(wav_x)
        # Add EPS to avoid zeros.
        original_max_psd = torch.as_tensor(original_max_psd)\
                                            .unsqueeze(0).to(self.device) + EPS
        theta = torch.as_tensor(theta).to(self.device) + EPS

        # c_mk = ECF * pow(10, 9.6) / (original_max_psd.t() * theta) / (self.audi.win_length ** 2)
        c_mk = ECF * torch.pow(torch.tensor(10.0).type(torch.float64), 
                         torch.tensor(9.6).type(torch.float64)).to(self.device) \
                / (original_max_psd.t().type(torch.float64) * theta.type(torch.float64)) \
                / (self.audi.win_length ** 2)
        # print("c_mk+++++++++++++++++++", c_mk)
        
        if info:
            print(f'theta: {list(theta.size())}\t max: {theta.max().item()}\t min: {theta.min().item()}')
            print(f'maxpad: {list(original_max_psd.size())}\t max: {original_max_psd.max().item()}\t min: {original_max_psd.min().item()}')
            print(f'c_mk: {list(c_mk.size())}\t max: {c_mk.max().item()}\t min: {c_mk.min().item()}')
        
        return c_mk
    

    def certify(self, x: torch.Tensor, n0: int, n: int, alpha: float, batch_size: int, info: bool = False) -> Tuple[(int, float)]:
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L_PSY radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L_PSY ball of radius R around x.

        Arguments
        ---------
        x : torch.Tensor
            The input [1, n_points].
        n0 : int
            The number of Monte Carlo samples to use for selection.
        n : int
            The number of Monte Carlo samples to use for estimation.
        alpha : float
            The failure probability.
        batch_size : int
            Batch size to use when evaluating the base classifier.

        Returns
        -------
        (predicted class, certified radius)
            in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection, _ = self._sample_noise(x, n0, batch_size, info)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation, c_mk_size = self._sample_noise(x, n, batch_size, info)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)

        if info:
            print('--> counts_selection:\t', counts_selection)
            print('--> cAhat:\t', cAHat)
            print('--> counts_estimation:\t', counts_estimation)
            print('--> pABar:\t', pABar)
        if pABar < 0.5:
            return Vmoat.ABSTAIN, 0.0
        else:
            # The formula of certified radius.
            radius = (self.sigma * norm.ppf(pABar))**2 / (c_mk_size[0]*c_mk_size[1])
            # print('c_mk_size[0]*c_mk_size[1]', c_mk_size[0], c_mk_size[1])
            return cAHat, radius


    def predict(self, x: torch.Tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        Arguments
        ---------        
        x : torch.Tensor 
            The input [channel x height x width].
        n : int
            The number of Monte Carlo samples to use.
        alpha : float
            The failure probability.
        batch_size : int
            Batch size to use when evaluating the base classifier.
        
        Returns
        -------
            The predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts, _ = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Vmoat.ABSTAIN
        else:
            return top2[0]
        
    def predict_to_be_attack(self, x: torch.Tensor, n: int, batch_size: int) -> torch.Tensor:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        Arguments
        ---------        
        x : torch.Tensor 
            The input [channel x height x width].
        n : int
            The number of Monte Carlo samples to use.
        alpha : float
            The failure probability.
        batch_size : int
            Batch size to use when evaluating the base classifier.
        
        Returns
        -------
            The predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        scores, _ = self._sample_noise_soft(x, n, batch_size)
        return scores / n

    # def _presample_noise(self, wav_x, batchsize): #【before】
    def _presample_noise(self, c_mk: torch.Tensor, batchsize: int, info: bool=True) -> torch.Tensor:
        ''' Obtain the time-domain noise.

        It first samples a batch of random spectrograms, and transform them into
        time-domain noises via utils.v_istft().

        Arguments
        ---------  
        c_mk : torch.Tensor
            The coefficients computed by self._noise_dist().
        batchsize : int
            The number of noises to generate.

        Returns
        -------
        noise : torch.Tensor
            The noise generated from utils.v_istft().
        '''
    
        c_mk_batch = c_mk.unsqueeze(0).repeat((batchsize, 1, 1))

        Delta_real = torch.randn_like(c_mk_batch) / torch.sqrt(c_mk) * self.sigma
        Delta_imag = torch.randn_like(c_mk_batch) / torch.sqrt(c_mk) * self.sigma

        noise = utils.v_istft(Delta_real, Delta_imag, original_win_length=2048, padding=PADDING)

        if info:
            print(f'noise: {list(noise.size())}, {noise.real.max().item()}, {noise.real.min().item()}')
            print('Overflow ratio', ((noise.real>1) | (noise.real<-1)).sum()/noise.size(1)/noise.size(2)/noise.size(0))
        
        return torch.real(noise).type(torch.float32)
    
    def batch_presample_noise(self, c_mk_batch: torch.Tensor, info: bool=True) -> torch.Tensor:
        ''' Obtain the time-domain noise.

        It first samples a batch of random spectrograms, and transform them into
        time-domain noises via utils.v_istft().

        Arguments
        ---------  
        c_mk : torch.Tensor
            The coefficients computed by self._noise_dist().
        batchsize : int
            The number of noises to generate.

        Returns
        -------
        noise : torch.Tensor
            The noise generated from utils.v_istft().
        '''
    
        # c_mk_batch = c_mk.unsqueeze(0).repeat((batchsize, 1, 1))

        Delta_real = torch.randn_like(c_mk_batch) / torch.sqrt(c_mk_batch) * self.sigma
        Delta_imag = torch.randn_like(c_mk_batch) / torch.sqrt(c_mk_batch) * self.sigma

        noise = utils.v_istft(Delta_real, Delta_imag, original_win_length=2048, padding=PADDING)

        if info:
            print(f'noise: {list(noise.size())}, {noise.real.max().item()}, {noise.real.min().item()}')
            print('Overflow ratio', ((noise.real>1) | (noise.real<-1)).sum()/noise.size(1)/noise.size(2)/noise.size(0))
        
        return torch.real(noise).type(torch.float16)

    def zero_grad(self):
        self.base_classifier.zero_grad()


    def _sample_noise_soft(self, x: torch.Tensor, num: int, batch_size) -> Tuple[(torch.Tensor, List)]:
        ''' Sample the base classifier's prediction under noisy corruptions of the input x.

        Arguments
        ---------
        x : torch.Tensor
            The input wav of size [1, n_points].
        num : int
            Number of samples/predictions to collect.
        batch_size : int
            Batch size for inference.
        
        Returns
        -------
        counts : np.ndarray[int]
            An ndarray[int] of length num_classes containing the per-class counts.
        '''
        counts = torch.zeros([1, self.num_classes]).to(self.device)

        # Trim the input wav instead of padding.
        tail = x.shape[1] % self.audi.win_length
        if tail != 0:
            trim_x = x[:, :-tail]
        else:
            trim_x = x

        c_mk = self._noise_dist(trim_x.detach(), False)

        for _ in (range(ceil(num / batch_size))):                            
            this_batch_size = min(batch_size, num)                         
            num -= this_batch_size
            torch.random.manual_seed(1234)
            noise = self._presample_noise(c_mk, this_batch_size, False).to(self.device)
            
            noise = noise.squeeze(1)

            input_batch = trim_x + noise
            # input_batch = trim_x + noise

            # print('Overflow ratio', ((input_batch>1) | (input_batch<-1)).sum()/input_batch.size(1)/input_batch.size(0))
            
            # Smoothed predictions
            predictions = self.base_classifier(input_batch.clip(min = -1, max = 1))
            counts += predictions.sum(dim=0, keepdim=True)

        return counts, list(c_mk.size())
    



    def _sample_noise(self, x: torch.Tensor, num: int, batch_size, info: bool = False, feed_cmk=False) -> Tuple[(np.ndarray, List)]:
        ''' Sample the base classifier's prediction under noisy corruptions of the input x.

        Arguments
        ---------
        x : torch.Tensor
            The input wav of size [1, n_points].
        num : int
            Number of samples/predictions to collect.
        batch_size : int
            Batch size for inference.
        
        Returns
        -------
        counts : np.ndarray[int]
            An ndarray[int] of length num_classes containing the per-class counts.
        '''
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)

            # Trim the input wav instead of padding.
            tail = x.shape[1] % self.audi.win_length
            if tail != 0:
                trim_x = x[:, :-tail]
            else:
                trim_x = x

            c_mk = self._noise_dist(trim_x, info)

            if info:
                pbar = tqdm(range(ceil(num / batch_size)))
            else:
                pbar = (range(ceil(num / batch_size)))

            for _ in pbar:                            
                this_batch_size = min(batch_size, num)                         
                num -= this_batch_size

                noise = self._presample_noise(c_mk, this_batch_size, info).to(self.device)
                noise = noise.squeeze(1)

                input_batch = trim_x + noise
                # print('Overflow ratio', ((input_batch>1) | (input_batch<-1)).sum()/input_batch.size(1)/input_batch.size(0))
                # torchaudio.save('Test_vmoat_sigma_1.wav', (input_batch/input_batch.abs().max())[0:1, :].cpu().type(torch.float32), 16000)
                # exit()
                # Smoothed predictions
                if not feed_cmk:
                    predictions = self.base_classifier(input_batch.clip(min = -1, max = 1))
                else:
                    predictions = self.base_classifier(input_batch.clip(min = -1, max = 1), c_mk.unsqueeze(0))
                # predictions = self.base_classifier(input_batch/input_batch.abs().max())
                
                predictions = predictions.argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)

            return counts, list(c_mk.size())
        

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        ''' Count the number of each class.

        Arguments
        ---------
        arr : np.ndarray
            Predictions computed by the base classifier, e.g., [0, 0, 1, 0, ..., 1].
        length : int
            The number of classes.
        
        Returns
        -------
        counts : np.ndarray[int]
            An ndarray[int] of length num_classes containing the per-class counts.
        '''
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts


    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        Arguments
        ---------
        NA : int
            The number of "successes".
        N : int
            The number of total draws.
        alpha : float
            The confidence level.

        Returns
        -------
            A lower bound on the binomial proportion which holds true 
            w.p at least (1 - alpha) over the samples.
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]



if __name__ == '__main__':
    import sys
    sys.path.append('/home/vmoat/vmoat_code')
    from model.sb_interfaces import Verification

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    audi = Audibility(
                    sample_rate = 16000,
                    win_length = 2048,
                    hop_length = 2048,
                    n_fft = 2048,
                    device = device,
                    mode='vmoat'
                    )

    verification = Verification.from_hparams(
                                source="speechbrain/spkrec-ecapa-voxceleb",
                                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                run_opts = {
                                    "device": device,
                                    "data_parallel_backend": True,
                                })
    verification.eval()

    # wav_file = '/home/vmoat/shibo/Code/paper1_random_smooth/smoothing-master/code/spk1_snt1.wav'
    wav_file = "/home/vmoat/LibriSpeech/test-clean/61/70968/61-70968-0002.flac"

    wav_x, sr_x = torchaudio.load(wav_file)
    wav_x = wav_x[:, :-(wav_x.shape[1] % 2048)]

    wav_delta = torch.randn_like(wav_x)/100
    # adv_wav_x, _ = torchaudio.load('adv_wav_x.wav')
    # wav_delta = adv_wav_x - wav_x
    # wav_delta = torch.clamp(wav_delta + wav_x, min=-1, max=1) - wav_x

    temp_a = audi.audibility(wav_delta.unsqueeze(0).to(device), wav_x)
    
    print(audi.mode, 'trim', temp_a[1].item())



    vmoat = Vmoat(verification, num_classes = 2, sigma = 1e2)
    c_mk = vmoat._noise_dist(wav_x).to(device)
    noise = utils.v_stft(wav_delta, original_win_length=2048, padding=PADDING).to(device)

    # This result should be the same as that output by audi.audibility().
    print((c_mk*(noise.real**2+noise.imag**2)).mean().item())
    print('noise.size()', noise.size())
