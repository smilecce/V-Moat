import torch
import math

def v_stft(wav_x: torch.Tensor, original_win_length: int, padding: int=0):
    ''' Transform the time-domain signal to complex spectrogram. Retangle window
    function and zero overlap are used for FFT.

    Arguments
    ---------
    wav_x : torch.Tensor
        The wav should be of size (1, n_points) or (batch, 1, n_points).
    original_win_length : int
        The window length to perform STFT, e.g., 2048.
    padding : int
        Pad each frame with zeros symmetrically, e.g., 0.

    Returns
    -------
    transformed_x : torch.Tensor
        The STFT of wav_x of size (batch, n_frames, n_fft//2+1).
    '''
    if len(wav_x.size()) == 2:  # If there is no batch dimension
        wav_x.unsqueeze(0)

    assert wav_x.size(-1) % original_win_length == 0, \
        f'''Input should be trimmed to satisfying 'n_points 
            ({wav_x.size(2)}) % win_length ({original_win_length}) == 0'
            '''
    # padding
    # leng = (original_win_length - (wav_x.size(-1) % original_win_length)) \
    #         if wav_x.size(-1) % original_win_length!=0  \
    #         else 0
    # wav_x = torch.nn.functional.pad(wav_x, (0, leng))

    # Reshape and pad reduntant 0
    frames_x = wav_x.reshape(wav_x.size(0), -1, original_win_length)
    frames_x_pad = torch.nn.functional.pad(frames_x, (padding//2, padding//2))

    window_fn = torch.hamming_window(window_length=original_win_length+padding, periodic=True)
    frames_x_pad *= window_fn.to(frames_x_pad.device)

    transformed_x = torch.fft.fft(frames_x_pad, 
                                  n=None, 
                                  dim=-1, 
                                  norm='backward')  # 1/N is multipled here.

    # Return half of the spectrograms.
    transformed_x = transformed_x[..., :transformed_x.size(2)//2 + 1]
    
    if len(wav_x.size()) == 2:  # If there is no batch dimension
        transformed_x = transformed_x.squeeze(0)

    return transformed_x


def v_istft(sampled_real, sampled_imag, original_win_length, padding=0):
    ''' Transfrom spectrogram into time-domain signal. The input sepctrogram is
    specified by its real and imag part separately.

    Arguments
    ---------
    sampled_real : torch.Tensor
        The real part of the spectrogram of size (batch, n_frames, n_bins).
    sampled_imag : torch.Tensor
        The imaginary part of the spectrogram of size (batch, n_frames, n_bins).
    original_win_length : int
        The window length to perform STFT, e.g., 2048.
    padding : int
        Pad each frame with zeros symmetrically, e.g., 0.

    Returns
    -------
    noise : torch.Tensor
        The corresponding time-domain noises of size (batch, 1, n_points).
    '''
    Delta_real = torch.cat(
                    [sampled_real, torch.flip(sampled_real[:, :, 1:-1], dims=[-1])],
                    dim=-1
                    )
    Delta_imag = torch.cat(
                    [sampled_imag, torch.flip(-sampled_imag[:, :, 1:-1], dims=[-1])],
                    dim=-1
                    )
    Delta = Delta_real + 1.j * Delta_imag  
    delta_mn = torch.fft.ifft(Delta,
                              n=None,
                              dim=-1,
                              norm='backward')
    
    window_fn = torch.hamming_window(window_length=original_win_length+padding, periodic=True)
    delta_mn /= window_fn.to(delta_mn.device)

    if padding != 0:
        delta_mn = delta_mn[:, :, padding//2:-padding//2]
    delta_mn = delta_mn.reshape(sampled_real.size(0), 1, -1)

    return delta_mn



def v_istft_bak(sampled_real, sampled_imag, original_win_length, padding=32):
    '''
    The frequency domain is transformed to the time domain.

    ::params:   sampled_real:   [batch, n_frames, n_bins]
                sampled_imag:   [batch, n_frames, n_bins]
    ::return:   noise:  [batch, n]
    '''
    device = sampled_real.device
    win_length = original_win_length + padding

    Delta_real = torch.cat(
                    [sampled_real, torch.flip(sampled_real[:, :, 1:-1], dims=[-1])],
                    dim=-1
                    )
    Delta_imag = torch.cat(
                    [sampled_imag, torch.flip(-sampled_imag[:, :, 1:-1], dims=[-1])],
                    dim=-1
                    )
    
    k = torch.arange(0, win_length).unsqueeze(0)
    n = torch.arange(0, win_length).unsqueeze(0)
    matrix_kn = 1.0 * torch.matmul(k.T, n).to(device)

    window = torch.hann_window(win_length + 2, periodic = True)[1:-1].to(device) 

    Delta_full = Delta_real + 1.j * Delta_imag  
    e_j = torch.exp(1.j * 2. * math.pi * matrix_kn / win_length)

    Delta_full = Delta_full.type(torch.complex128)  #.to(device)
    e_j = e_j.type(torch.complex128)                #.to(device)

    delta_mn1 = torch.matmul(Delta_full, e_j) / (win_length)                  #.to(device)
    
    delta_mn1 = (delta_mn1 / window)[:, :, padding//2:-padding//2]
    # delta_mn1 = (delta_mn1)[:, :, padding//2:-padding//2]
    delta_mn1 = delta_mn1.reshape(sampled_real.size(0), 1, -1)

    return delta_mn1

def v_stft_bak(wav_x, original_win_length, padding=32):
    '''
    Test: transform the time-domain signal to spectral representation.

    ::param: wav_x      Tensor, [1, n_points] or [batch, 1, n_points]
    '''
    device = wav_x.device

    if len(wav_x.size()) == 2:  # If there is no batch dimension
        wav_x.unsqueeze(0)

    win_length = original_win_length + padding
    window_fn = torch.hann_window(win_length + 2, periodic = True)[1:-1].to(device)  # type: ignore

    # padding
    leng = (original_win_length - (wav_x.size(-1) % original_win_length)) \
            if wav_x.size(-1) % original_win_length!=0  \
            else 0
    wav_x = torch.nn.functional.pad(wav_x, (0, leng))


    frames_x = wav_x.reshape(wav_x.size(0), -1, original_win_length)
    frames_x_pad = torch.nn.functional.pad(frames_x, (padding//2, padding//2))

    k = torch.arange(0, win_length).unsqueeze(0)
    n = torch.arange(0, win_length).unsqueeze(0)

    matrix_kn = 1. * torch.matmul(k.T, n).to(device)
    e_j_inv = torch.exp(- 1.j * 2. * math.pi * matrix_kn / win_length)

    e_j_inv = e_j_inv.type(torch.complex128)                #.to(device)
    frames_x_pad = frames_x_pad.type(torch.complex128)      #.to(device)
    window_fn = window_fn.type(torch.complex128)            #.to(device)

    frames_x_new = frames_x_pad * window_fn
    transformed_x = torch.matmul(frames_x_new, e_j_inv)

    # [n_frames, win_length = 2080]
    transformed_x = transformed_x[..., :transformed_x.size(2)//2 + 1]
    
    if len(wav_x.size()) == 2:  # If there is no batch dimension
        transformed_x = transformed_x.squeeze(0)

    return transformed_x

def istft_basic(X, w, H, L):
    """Compute the inverse of the basic discrete short-time Fourier transform (ISTFT)
    Args:
        X (np.ndarray): The discrete short-time Fourier transform
        w (np.ndarray): Window function
        H (int): Hopsize
        L (int): Length of time signal
    Returns:
        x (np.ndarray): Time signal
    """
    import numpy as np

    N = len(w)
    M = X.shape[1]
    x_win_sum = np.zeros(L)
    w_sum = np.zeros(L)
    for m in range(M):
        x_win = np.fft.ifft(X[:, m])
        x_win = np.real(x_win) * w
        x_win_sum[m * H:m * H + N] = x_win_sum[m * H:m * H + N] + x_win
        w_shifted = np.zeros(L)
        w_shifted[m * H:m * H + N] = w * w
        w_sum = w_sum + w_shifted
    # Avoid division by zero
    w_sum[w_sum == 0] = np.finfo(np.float32).eps
    x_rec = x_win_sum / w_sum
    return x_rec, x_win_sum, w_sum

def stft_basic(x, w, H=8, only_positive_frequencies=False):
    """Compute a basic version of the discrete short-time Fourier transform (STFT)
    Args:
        x (np.ndarray): Signal to be transformed
        w (np.ndarray): Window function
        H (int): Hopsize
        only_positive_frequencies (bool): Return only positive frequency part of spectrum (non-invertible)
            (Default value = False)
    Returns:
        X (np.ndarray): The discrete short-time Fourier transform
    """
    import numpy as np

    N = len(w)
    L = len(x)
    M = np.floor((L - N) / H).astype(int) + 1
    X = np.zeros((N, M), dtype='complex')
    for m in range(M):
        x_win = x[m * H:m * H + N] * w
        X_win = np.fft.fft(x_win)
        X[:, m] = X_win

    if only_positive_frequencies:
        K = 1 + N // 2
        X = X[0:K, :]
    return X



if __name__ == '__main__':
    noise = torch.randn([10, 1, 20480]).clip(min=-1, max=1)
    transform_noise = v_stft(noise, 2048, padding=0)
    # print(transform_noise.size())   # torch.Size([10, 10, 1025])

    sampled_real = torch.randn_like(transform_noise.real) / 20 * 1
    sampled_imag = torch.randn_like(transform_noise.real) / 20 * 1

    noise = v_istft(sampled_real, sampled_imag, original_win_length=2048, padding=0)
    transform_noise = v_stft(noise, 2048, padding=0)

    print(noise.shape, transform_noise.shape)
    print(((sampled_real-transform_noise.real)/sampled_real).abs().mean())
    print(((sampled_imag-transform_noise.imag)/sampled_imag).abs().mean())