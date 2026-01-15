This repository provides the official implementation of **V-Moat**. It includes the framework for model enhancement via multi-sigma fine-tuning, certified robustness evaluation, and human audibility analysis.

## 🚀 Getting Started

### Installation

Clone the repository and set up the environment using Conda:

Bash

```
git clone https://github.com/smilecce/V-Moat.git
cd V-Moat

conda create -n vmoat python=3.9
conda activate vmoat
pip install -r requirements.txt
```

### Dataset Preparation

Our experiments utilize the following datasets:

- **Speaker Recognition:** [VoxCeleb1 & VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/), [CN-Celeb](https://openslr.org/82/), [LibriSpeech](https://www.openslr.org/12/)
- **SDR:** [Spoken Numbers PCM](https://github.com/pannous/tensorflow-speech-recognition)

Please ensure datasets are downloaded and organized according to the paths specified in the configuration files.

------

## 🛠️ Model Enhancement

To improve model robustness, we employ fine-tuning under varying noise scales. You can initiate the fine-tuning process using specific configuration files for each $\sigma$ value:

```
CUDA_VISIBLE_DEVICES=0,1 python train_speaker_embeddings.py \
    hparams/train_ecapa_tdnn_sigma_10.yaml \
    --data_parallel_backend
```

*Note: Choose the appropriate YAML configuration based on your target task and noise levels.*

------

## 🛡️ Robustness Certification

###  Smoothed Classifiers

The Vmoat is implemented in the `Vmoat` class within `vmoat.py`. This class transforms a standard base classifier into a robust **smoothed classifier** $g$.

#### **Core Implementation & Dependencies**

The robustness of V-Moat relies on several key components:

- **Audibility Calculation (`certification/audibility.py`)**: Defines the Audibility Metric $\mathcal{A}$ used to bound noise perceptibility. Optimize the audibility metric parameters based on collected human responses using contrastive learning, `userstudy/contrastive_learn_fit3.py`
- **Psychoacoustic Thresholding (`certification/masker.py`)**: Implements the calculation of the masking threshold  based on frequency-domain masking effects.
- **Signal Processing (`certification/utils.py`)**: Contains optimized STFT and ISTFT functions used to transition between time-domain audio and the frequency-domain masking model.

#### **Instantiation**

To instantiate a smoothed classifier $\mathcal{G}$, use the constructor:

```
def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
```

- `base_classifier`: A PyTorch module that implements the base model $f$.
- `num_classes`: The number of classes in the output space.
- `sigma`: The noise hyperparameter, controlling the smoothing variance.

#### **Prediction**

To make a robust prediction for an input audio clip $x$, call:

```
def predict(self, x: torch.Tensor, n: int, alpha: float, batch_size: int) -> int:
```

- `n`: The number of Monte Carlo samples.
- `alpha`: The failure probability (confidence level).
- **Returns**: Either `-1` (abstain) or a class label that equals $g(x)$ with probability at least $1 - \alpha$.

#### **Certification**

To compute the certified radius within which g 's prediction is guaranteed to be constant around an input x :

```
def certify(self, x: torch.Tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
```

- `n0`: Number of Monte Carlo samples for initial selection (top-class hypothesis).
- `n`: Number of Monte Carlo samples for estimation.
- `alpha`: The confidence level.
- **Returns**: A pair `(prediction, radius)`. If the algorithm abstains, it returns `(-1, 0.0)`. With probability at least $1 - \alpha$, the returned prediction equals $\mathcal{G}$, and the model is robust within the certified  radius.

### Certified Accuracy

To evaluate the certified robustness of VMoat, use the following script to calculate and plot the Certified Accuracy :

```
python certify_plot_accuracy_radius.py \
    --model_name FT33Sigmafix_30 \
    --sigma 30 \
    --batch 120 \
    --N0 100 \
    --N 100000 \
    --alpha 0.001 \
    --model_src ./model/trained_models/CKPT_PATH
```

### Empirical Bounds

To calculate the empirical audibility bounds and optimize detection thresholds:

```
python empirical_bound_save_best_audi.py \
    --sigma 10 \
    --batch 120 \
    --model_src ./model/trained_models/CKPT_PATH
```

------

## 👥 User Study & Perceptual Analysis

We conduct user studies to validate the alignment between our proposed Audibility Metric ($\mathcal{A}$) and human perception. All studies are approved by our Institutional Review Board (IRB).

Use `Questionnaire.py` to generate perceptual test samples. Optimize the audibility metric parameters based on collected human responses using contrastive learning

To evaluate whether there is an overlap between the noise audibility and human perception, we use `survey_test.py`  to collect data. The detailed statistical analysis and visualization of user performance are provided in:`answers_analysis.ipynb`

------

## ⚖️ License

This project is released under the [MIT License](https://www.google.com/search?q=LICENSE).
