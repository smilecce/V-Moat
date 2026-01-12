from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import Pretrained
from speechbrain.pretrained.interfaces import foreign_class

import torchaudio
import torch
import torch.nn as nn
from torch.nn import Softmax

from typing import Dict, List, Tuple

import math
import torchvision.transforms.functional as F
import torchaudio.transforms as T

import math
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from librosa import display
import librosa
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Verification(SpeakerRecognition):

    MODULES_NEEDED = [
    "compute_features",
    "mean_var_norm",
    "embedding_model",
    "mean_var_norm_emb",
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enrolled_emb = None
        # The threshold is fixed to 0.25.
        self.threshold = torch.FloatTensor([0.25]).to(self.device)      # (1,)
        self.softmax = Softmax(dim=-1)

    def enroll(self, enrolled_wav: torch.Tensor):
        ''' Enroll one audio to be verified.

        Arguments
        ---------        
        enrolled_wav : torch.Tensor
            The size should be [1, n_points].
        '''
        assert enrolled_wav.size(0) == 1
        self.enrolled_emb = self.encode_batch(enrolled_wav, wav_lens=None, normalize=False)

        return self.enrolled_emb
       

    def forward(self, wavs: torch.Tensor, wav_lens=None):
        ''' Forward for verification.

        It first encodes the wavs into embeddings and then computes the similarities
        between the trials and the enrollment audio.

        Arguments
        ---------  
        wavs : torch.Tensor
            The size should be [batch, n_points].
        
        Returns
        -------
        p : torch.Tensor
            The probability distribution with a size of [batch, 2].
        '''

        emb = self.encode_batch(wavs, wav_lens, normalize=False)

        score = self.similarity(emb, self.enrolled_emb)     # score: torch.Size([2, 1])

        # Concat the scores and the thresholds (repeated) to fake a binary classification.
        scores = torch.hstack([score, self.threshold.repeat(score.size(0), 1)])     # scores: (batch, 2)
        p = self.softmax(scores) # p: (batch, 2)

        return p


class Classification(EncoderClassifier):
    ''' Closed-set identification
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.enrolled_spks = {}
        self.spkid_list =[]
        self.softmax = Softmax(dim=-1)

    def enroll_many(self, enroll_list_dict: List[Dict]):
        ''' Enroll multiple speakers with an enroll_dict.

        It reads the items in enroll_list_dict and enrolls each speaker with
        the 'wav' and 'spk_id' values in each dict.

        Arguments
        ---------
        enroll_list_dict : List[Dict]
            A list containing dicts for enrollment.
            e.g., enroll_list_dict = [{
                            'wav': torch.Tensor,
                            'spk_id': str
                        }]
        '''
        for speaker in  enroll_list_dict:
            self.enroll(
                    wav_path=speaker['wav_path'], 
                    spk_id=speaker['spk_id']
                    )
        
        # Print a summary of enrolled speakers.
        print('Enrollment finished. The enrolled speakers are as follows:')
        for idx, spk_id in enumerate(self.enrolled_spks):
            print(f'Speaker {idx}: {spk_id}')


    def enroll(self, wav_path: str, spk_id: str):
        ''' Enroll one audio for one speaker.

        It loads the wav file according to the wav_path, encodes the wav into
        an embedding and stores it in a dict.

        Arguments
        ---------
        wav_path : str
            The path to the enrollment audio
        spk_id : str
            The ID of the current speaker
        '''
        enrolled_wav, sr = torchaudio.load(wav_path)

        # Compute the embedding of the current wav.
        self.enrolled_spks[spk_id] = self.encode_batch(
            enrolled_wav.to(self.device), wav_lens=None, normalize=False)


    def index_decode_2spk_id(self, spk_ind: int):
        ''' Return the speaker id given the index of the speaker (output of 
        self.classify_batch).
        '''
        self.spkid_list = list(self.enrolled_spks.keys())
        return self.spkid_list[spk_ind]

    def classify_batch(self, wavs: torch.Tensor, wav_lens: torch.Tensor=None):
        """ Performs classification on the top of the encoded features.

        It returns the posterior probabilities, the index and, if the label
        encoder is specified it also the text label.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        out_prob
            The log posterior probabilities of each class ([batch, N_class])
        score:
            It is the value of the log-posterior for the best class ([batch,])
        index
            The indexes of the best class ([batch,])
        text_lab:
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        # Encode the input wavs into embeddings.
        input_emb = self.encode_batch(wavs, wav_lens)
        
        score_list = []         # Collect the similarities between each input wav and
                                # enrollment audios of each speakers.

        for spk_id, emb in self.enrolled_spks.items():
            temp_score = self.similarity(emb, input_emb)
            score_list.append(temp_score)

        # score_tensor: [batch, n_classes]
        score_tensor = torch.cat(score_list, dim=1)

        # print('[score_tensor]\n', score_tensor)
        return score_tensor


    def num_classes(self):
        ''' Return the number of speakers enrolled.
        '''
        self.class_num = len(self.enrolled_spks)
        
        return self.class_num
    
    def forward(self, wavs: torch.Tensor, wav_lens: torch.Tensor=None):
        """ Performs classification on the top of the encoded features.

        It returns the posterior probabilities, the index and, if the label
        encoder is specified it also the text label.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        out_prob
            The log posterior probabilities of each class ([batch, N_class])
        score:
            It is the value of the log-posterior for the best class ([batch,])
        index
            The indexes of the best class ([batch,])
        text_lab:
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        return self.classify_batch(wavs, wav_lens)
        # 继承的那里，返回的是，所有的分数，最好的的分数，最好的分数对应的index，最好的分数对应的index的speaker_id。
    
    
class Verification_CMK(SpeakerRecognition):

    MODULES_NEEDED = [
    "compute_features",
    "mean_var_norm",
    "embedding_model",
    "mean_var_norm_emb",
    "cmk_asist"
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enrolled_emb = None
        # The threshold is fixed to 0.25.
        self.threshold = torch.FloatTensor([0.25]).to(self.device)      # (1,)
        self.softmax = Softmax(dim=-1)

    def enroll(self, enrolled_wav: torch.Tensor):
        ''' Enroll one audio to be verified.

        Arguments
        ---------        
        enrolled_wav : torch.Tensor
            The size should be [1, n_points].
        '''
        assert enrolled_wav.size(0) == 1
        self.enrolled_emb = self.encode_batch(enrolled_wav, wav_lens=None, normalize=False)

        return self.enrolled_emb
       

    def forward(self, wavs: torch.Tensor, cmk=None, wav_lens=None):
        ''' Forward for verification.

        It first encodes the wavs into embeddings and then computes the similarities
        between the trials and the enrollment audio.

        Arguments
        ---------  
        wavs : torch.Tensor
            The size should be [batch, n_points].
        
        Returns
        -------
        p : torch.Tensor
            The probability distribution with a size of [batch, 2].
        '''

        emb = self.encode_batch(wavs, cmk, wav_lens, normalize=False)

        score = self.similarity(emb, self.enrolled_emb)     # score: torch.Size([2, 1])

        # Concat the scores and the thresholds (repeated) to fake a binary classification.
        scores = torch.hstack([score, self.threshold.repeat(score.size(0), 1)])     # scores: (batch, 2)
        p = self.softmax(scores) # p: (batch, 2)

        return p
    
    def encode_batch(self, wavs, cmk=None, wav_lens=None, normalize=False):
        """Encodes the input audio into a single vector embedding.

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = <this>.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        normalize : bool
            If True, it normalizes the embeddings with the statistics
            contained in mean_var_norm_emb.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()



        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)

        if cmk == None:
            c_mk_rs = torch.zeros_like(feats)
        else:
            c_mk_rs = 1 / torch.sqrt(F.resize(cmk, (feats.size(1), feats.size(2))))


        cmk_feats = self.mods.cmk_asist(c_mk_rs)
        feats = feats * (1 + cmk_feats)
        
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings
    
class emotion_class(Pretrained):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, wav_x):
        classifier = foreign_class(
                            # source="/home/vmoat/shibo/Code/speechbrain/recipes/IEMOCAP/emotion_recognition/pretrained_models/CustomEncoderWav2vec2Classifier-6d849d67d0260d5a3ea6b156bb50a54b",
                            source="/home/vmoat/shibo/Code/speechbrain/recipes/IEMOCAP/emotion_recognition/hubert_large/1993/results/hubert_large_fold1_exp1/1993/save/CKPT+2023-08-18+17-12-02+00",
                            # source="/home/vmoat/shibo/Code/speechbrain/recipes/IEMOCAP/emotion_recognition/hubert_large/1993/results/hubert_large_fold1_exp1/1993/save/CKPT+2023-08-15+03-42-19+00",
                            hparams_file='test_hyperparams.yaml',
                            pymodule_file="custom_interface.py", 
                            classname="CustomEncoderWav2vec2Classifier"
                            )
        print('【forward】')
    # # ret = classifier.classify_file("speechbrain/emotion-recognition-wav2vec2-IEMOCAP/anger.wav")
        # wav_path = "/home/vmoat/shibo/Code/speechbrain/recipes/IEMOCAP/emotion_recognition/anger.wav"
        out_prob, score, index, text_lab = classifier.classify_batch(wav_x)    # # score: torch.Size([2, 1])
        print('【text_lab】', text_lab)
        # print(out_prob)
        # print(out_prob.shape)

        # scores = torch.hstack([out_prob, threshold.repeat(out_prob.size(0), 1)])     # scores: (batch, 2)
        # print(scores)
        softmax = torch.nn.Softmax(dim=1)
        p = softmax(out_prob)
        print('【p】', p)
        return p
    



class CustomEncoderWav2vec2Classifier(Pretrained):
    """A ready-to-use class for utterance-level classification (e.g, speaker-id,
    language-id, emotion recognition, keyword spotting, etc).

    The class assumes that an self-supervised encoder like wav2vec2/hubert and a classifier model
    are defined in the yaml file. If you want to
    convert the predicted index into a corresponding text label, please
    provide the path of the label_encoder in a variable called 'lab_encoder_file'
    within the yaml.

    The class can be used either to run only the encoder (encode_batch()) to
    extract embeddings or to run a classification step (classify_batch()).
    ```

    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.pretrained import EncoderClassifier
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> classifier = EncoderClassifier.from_hparams(
    ...     source="speechbrain/spkrec-ecapa-voxceleb",
    ...     savedir=tmpdir,
    ... )

    >>> # Compute embeddings
    >>> signal, fs = torchaudio.load("samples/audio_samples/example1.wav")
    >>> embeddings =  classifier.encode_batch(signal)

    >>> # Classification
    >>> prediction =  classifier .classify_batch(signal)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        """Encodes the input audio into a single vector embedding.

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = <this>.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        normalize : bool
            If True, it normalizes the embeddings with the statistics
            contained in mean_var_norm_emb.

        Returns
        -------
        torch.tensor
            The encoded batch
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        outputs = self.mods.wav2vec2(wavs)

        # last dim will be used for AdaptativeAVG pool
        outputs = self.mods.avg_pool(outputs, wav_lens)
        outputs = outputs.view(outputs.shape[0], -1)
        return outputs

    def classify_batch(self, wavs, wav_lens=None):
        """Performs classification on the top of the encoded features.

        It returns the posterior probabilities, the index and, if the label
        encoder is specified it also the text label.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        out_prob
            The log posterior probabilities of each class ([batch, N_class])
        score:
            It is the value of the log-posterior for the best class ([batch,])
        index
            The indexes of the best class ([batch,])
        text_lab:
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        outputs = self.encode_batch(wavs, wav_lens)
        outputs = self.mods.output_mlp(outputs)
        out_prob = self.hparams.softmax(outputs)
        score, index = torch.max(out_prob, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab

    def classify_file(self, path):
        """Classifies the given audiofile into the given set of labels.

        Arguments
        ---------
        path : str
            Path to audio file to classify.

        Returns
        -------
        out_prob
            The log posterior probabilities of each class ([batch, N_class])
        score:
            It is the value of the log-posterior for the best class ([batch,])
        index
            The indexes of the best class ([batch,])
        text_lab:
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        waveform = self.load_audio(path)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        outputs = self.encode_batch(batch, rel_length)
        outputs = self.mods.output_mlp(outputs).squeeze(1)
        out_prob = self.hparams.softmax(outputs)
        score, index = torch.max(out_prob, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab

    def forward(self, wavs, wav_lens=None, normalize=False):
        return self.encode_batch(
            wavs=wavs, wav_lens=wav_lens, normalize=normalize
        )


class emotion_class2(CustomEncoderWav2vec2Classifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, wav_x):
        # classifier = foreign_class(
        #                     # source="/home/vmoat/shibo/Code/speechbrain/recipes/IEMOCAP/emotion_recognition/pretrained_models/CustomEncoderWav2vec2Classifier-6d849d67d0260d5a3ea6b156bb50a54b",
        #                     source="/home/vmoat/shibo/Code/speechbrain/recipes/IEMOCAP/emotion_recognition/hubert_large/1993/results/hubert_large_fold1_exp1/1993/save/CKPT+2023-08-18+17-12-02+00",
        #                     # source="/home/vmoat/shibo/Code/speechbrain/recipes/IEMOCAP/emotion_recognition/hubert_large/1993/results/hubert_large_fold1_exp1/1993/save/CKPT+2023-08-15+03-42-19+00",
        #                     hparams_file='test_hyperparams.yaml',
        #                     pymodule_file="custom_interface.py", 
        #                     classname="CustomEncoderWav2vec2Classifier"
        #                     )
        # print('【forward】')
    # # ret = classifier.classify_file("speechbrain/emotion-recognition-wav2vec2-IEMOCAP/anger.wav")
        # wav_path = "/home/vmoat/shibo/Code/speechbrain/recipes/IEMOCAP/emotion_recognition/anger.wav"
        out_prob, score, index, text_lab = self.classify_batch(wav_x)    # # score: torch.Size([2, 1])
        # print('【text_lab】', text_lab)
        # print(out_prob)
        print('out_prob', out_prob.shape)

        # scores = torch.hstack([out_prob, threshold.repeat(out_prob.size(0), 1)])     # scores: (batch, 2)
        # print(scores)
        # softmax = torch.nn.Softmax(dim=1)
        # p = softmax(out_prob)
        # print('【p】', p)
        return out_prob
    

class Spoken_digit_recognition(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_model = torch.load('/home/vmoat/spoken_numbers_pcm/Spokendigit-master/spokendigit_cnn_mel.pth')

    def forward(self, wav_x):

        data2 = wav_x
        data2 = data2.squeeze(0).numpy()

        fig = plt.figure(figsize=[0.774, 0.77], dpi=500)  # Set the DPI here
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

        S = librosa.feature.melspectrogram(y=data2, sr=sr)
        c = display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', fmin=50, fmax=280)

        # Convert the figure to a PyTorch tensor
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()

        # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = Image.fromarray(data)
        transform = transforms.Compose([transforms.ToTensor()])
        a = transform(img)
        
        a = a.to(device).unsqueeze(0)

        print(a.shape)

        probs = self.cnn_model(a)
        print(probs.shape)
        return probs


class SpokenDigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def validation_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        _, pred = torch.max(outputs, 1)
        accuracy = torch.tensor(torch.sum(pred==labels).item()/len(pred))
        return [loss.detach(), accuracy.detach()] 

class Spoken_digit_recognition2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.cnn_model = torch.load('/home/vmoat/spoken_numbers_pcm/Spokendigit-master/spokendigit_cnn_mel.pth')
        self.cnn_model = SpokenDigitModel().to(device)
        self.cnn_model.load_state_dict(torch.load('/home/vmoat/spoken_numbers_pcm/Spokendigit-master/MODEL/results/sigma_10/20230701/save/CKPT+2023-09-11+12-43-27+00/classifier.ckpt'))
        print('init-------')

    def forward(self, wav_x):

        input_a = torch.empty(0, 3, 385, 387).to(device)  #shapa is same with a
        # print('input_a]]]]]', input_a)

        wavs = wav_x
        # print('wav_x——shape', wavs.shape)
        for i in range(wavs.shape[0]):
            data2 = wavs[i,:]
            # print('data2.sahaoe', data2.shape)
            data2 = data2.squeeze(0).cpu().numpy()

            fig = plt.figure(figsize=[0.774, 0.77], dpi=500)  # Set the DPI here
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)

            S = librosa.feature.melspectrogram(y=data2) #, sr=sr)
            c = display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', fmin=50, fmax=280)

            # Convert the figure to a PyTorch tensor
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.canvas.draw()
            plt.close()  # 关闭Matplotlib图形

            # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = Image.fromarray(data)
            transform = transforms.Compose([transforms.ToTensor()])
            a = transform(img)
            
            a = a.to(device).unsqueeze(0)
            # print('a_shape', a.shape)
            # wavs = a
            # if input_a == None
            input_a = torch.cat((input_a, a), dim=0)
            # print('input_a]]]]]', input_a.shape)

        probs = self.cnn_model(input_a)
        # print(probs.shape)
        return probs
    
class Spoken_digit_recognition3(nn.Module):
    def __init__(self):
        super().__init__()
#       输入的size为（1，1,128,32）  (batch channel x，y)
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
#           有padding  1,16,128,32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),# kernel_size, stride
#             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
#             1 16 64 16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             1 32 64 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
#             1 32 32 8

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             1 64 32 8
            nn.ReLU(),
#             nn.MaxPool2d(2, 2),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
#             1 64 16 8
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
# #             1 128 16 8
            nn.ReLU(),
#             nn.MaxPool2d(2, 2),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             1 128 8 8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            # nn.Sigmoid()
        )
        self.mel_spectrogram = T.MelSpectrogram(
                sample_rate=16000, #sample_rate,
                n_fft=1024,
                win_length=1024,
                hop_length=512,
                center=True,
                pad_mode="constant",
                power=2.0,
                norm="slaney",
                n_mels=128,
                mel_scale="htk",
            ).to(device)
        self.transform_db_func = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)    
    def forward(self, x):
        if len(x.shape)==2:
            x = x.unsqueeze(1) #升一维，要让N作为batch的存在
        # print('【ori_x】', x.shape)

        x = x.to(device)
        melspec = self.mel_spectrogram(x)
        # transform_db_func = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        db_melspec = self.transform_db_func(melspec)
        # print('【db_shape】', db_melspec.shape)
        return self.network(db_melspec)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ==========================================================================================
    # Classification
    enroll_list_dict = [
                    {'wav_path':'/home/vmoat/LibriSpeech/test-clean/61/70968/61-70968-0000.flac','spk_id':'61'},
                    {'wav_path':'/home/vmoat/LibriSpeech/test-clean/121/121726/121-121726-0000.flac','spk_id':'121'},
                    {'wav_path':'/home/vmoat/LibriSpeech/test-clean/237/126133/237-126133-0000.flac','spk_id':'237'},
                    ]

    classification = Classification.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        run_opts = {
                            "device": device,
                            "data_parallel_backend": False,
                        })
    
    classification.eval()
    classification.enroll_many(enroll_list_dict)        #这里就是调用forward函数  

    path1 = "/home/vmoat/LibriSpeech/test-clean/61/70968/61-70968-0001.flac"
    path2 = "/home/vmoat/LibriSpeech/test-clean/121/121726/121-121726-0001.flac"
    wav1, sr = torchaudio.load(path1)
    wav2, sr = torchaudio.load(path2)

    p = classification(wav2.to(device))
    print('Classification (1):', p)

    p = classification(torch.vstack([wav1, wav1]).to(device))
    print('Classification (2) batch classification:\n', p)

    # ==========================================================================================
    # Verification
    verification = Verification.from_hparams(
                                    source="speechbrain/spkrec-ecapa-voxceleb",
                                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                    run_opts = {
                                        "device": device,
                                        "data_parallel_backend": False,
                                    })
    verification.eval()
    enrolled_wav = verification.enroll(wav1.to(device))

    # Call the forward function of the verfication，return 
    # the similarity of the 'input_wav' and the 'enrolled_wav'.
    p = verification(wav2.to(device))
    print('Verification (1):', p)

    p = verification(torch.vstack([wav1, wav1]).to(device))
    print('Verification (2) batch verification:\n', p)