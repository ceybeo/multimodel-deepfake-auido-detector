# AI-Generated Image Detector

A Streamlit app and training pipeline that estimates whether an image is a real photograph or AI-generated. It fine-tunes EfficientNet-B4 and, more importantly, evaluates it honestly: on held-out data sources and on independent test sets that were never used for tuning.

**Status:** the image module is complete. Video (frame-by-frame) and audio detection are planned and not implemented yet.

## Results

| Evaluation | Data | Result |
|---|---|---|
| Validation (held-out dataset shards, never seen in training) | 1,600 images | **91.4%** (Fake 91.4%, Real 91.4%) |
| Independent test, real photos | 19 photos from a phone not used in training | **94.7%** correct (18/19), 95% CI about 75-99% |
| Independent test, Gemini images | 31 new images | **80.6%** correct (25/31), AUC 0.90, 95% CI about 64-91% |
| Independent test, ChatGPT images (low resolution, median short side 242 px) | 11 images | **0/11** detected, AUC 0.62 |

The independent test sets are small, so treat these numbers as indicative, not precise. The ChatGPT images could only be saved as ~300 px previews; at that resolution the generator's traces are largely gone, so this result says little about full-size ChatGPT images.

## What I learned

The first model reached 99% training accuracy and labeled every real photo as fake. The reasons, found step by step:

1. **A labeling bug in my data preparation.** I derived labels from folder names, and every folder of one dataset contained the substring "ai", so real images were labeled fake. Training accuracy was measured against wrong labels.
2. **Old datasets do not generalize.** After fixing the labels, accuracy was still ~98%, but the model detected 0% of images from current generators (Gemini, ChatGPT). The datasets were built from older generators.
3. **Validation must come from unseen sources.** I switched to a newer dataset and kept whole shards out of training for validation. Accuracy dropped from 98% to 88%, which is the more honest number.
4. **File properties can be shortcuts.** My test images differed in resolution, file size and compression between classes. Every image is now resized and re-compressed identically before evaluation, and training uses random resizing and random JPEG compression so resolution cannot reveal the class.
5. **Real-photo false alarms.** Training data contained stock photos only; adding my own phone photos and class-weighted loss cut false alarms on real photos from about 26% to about 5% on my first test set, and the result held on a friend's phone (94.7%).

## Repository layout

```
app.py                Streamlit app
src/prepare_data.py   builds a balanced subset of the dataset
src/train.py          fine-tuning with resume support
src/evaluate.py       evaluation on independent folders (accuracy + CI, AUC, zones)
requirements.txt
```

## Installation

```
git clone https://github.com/ceybeo/<repo-folder>.git
cd <repo-folder>
pip install -r requirements.txt
```

## Usage

**App** (needs the trained weights, see below):

```
python -m streamlit run app.py
```

The app shows a probability, not a verdict: below 0.30 "likely real", above 0.70 "likely AI-generated", otherwise "uncertain". These thresholds are hand-picked, not tuned.

**Data preparation** (downloads about 58 GB of shards; the script deletes each shard after sampling):

```
python src/prepare_data.py --out xiz_veri
python src/prepare_data.py --out xiz_veri --own-real path/to/your/photos
```

**Training** (a GPU is strongly recommended; I used a Colab T4, about 5.5 min per epoch):

```
python src/train.py --data xiz_veri --out checkpoints
```

**Evaluation:**

```
python src/evaluate.py --model checkpoints/efficientnet_b4_best.pth --real path/to/real --fake gemini=path/to/gemini chatgpt=path/to/chatgpt --split-size 1000
```

## Data

Training data comes from [Xiz9Dataset](https://huggingface.co/datasets/KarmaLeo/Xiz9Dataset) on Hugging Face (about 14,000 images sampled), which combines the Kaggle dataset "AI-Generated Images vs Real Images" and the OpenFake dataset (ComplexDataLab). Its card states academic and research use only, so this project and its weights should be treated the same way. My own photos and the test images are not included in this repository, because they are personal photos.

## Limitations

- The test sets are small (11-31 images per group) and from few devices.
- Detection of generators not seen in training is partial (Gemini 80.6%).
- Very sharp, heavily processed real photos can be flagged as AI-generated. One of my 19 test photos was, with 0.99 fake probability.
- Low-resolution images (about 300 px) are not reliably classified.
- Image analysis only; no face-specific deepfake detection, no video, no audio.

## Model weights

The trained weights (about 70 MB) are published as a release asset:
[efficientnet_b4_v3.pth](https://github.com/ceybeo/multimodel-deepfake-auido-detector/releases/download/v0.1.0/efficientnet_b4_v3.pth)

Download the file and place it next to `app.py`. On Windows PowerShell:

```
Invoke-WebRequest -Uri https://github.com/ceybeo/multimodel-deepfake-auido-detector/releases/download/v0.1.0/efficientnet_b4_v3.pth -OutFile efficientnet_b4_v3.pth
```

Research use only (see the Data section).

## Roadmap

- [x] Image detector (EfficientNet-B4, Streamlit app, evaluation on independent test sets)
- [ ] Video: apply the image model frame by frame and aggregate the scores
- [ ] Audio: spectrogram-based detector for synthetic speech
- [ ] Larger and more diverse independent test sets (more devices, more generators, full-resolution ChatGPT images)
- [ ] Combined multimodal interface

## Author

Zeynep Özdemir, graduation project.