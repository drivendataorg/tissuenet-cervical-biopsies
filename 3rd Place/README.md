# TissueNet: Detect Lesions in Cervical Biopsies

[TissueNet: Detect Lesions in Cervical Biopsies](https://www.drivendata.org/competitions/67/competition-cervical-biopsy/).

[3rd place](https://www.drivendata.org/competitions/67/competition-cervical-biopsy/leaderboard/)
out of 547 with 0.9339 Weighted Class Score (top 1 -- 0.9475).

### Prerequisites

- [libvips](https://libvips.github.io/libvips/install.html)
- GPU with 32Gb RAM (e.g. Tesla V100)
- [NVIDIA apex](https://github.com/NVIDIA/apex)

```bash
pip install -r requirements.txt
```

### Usage

#### Download

First download the train data from the competition link into `data` folder.

Then you have to download train images

```bash
python ./src/download.py
```

This will download whole dataset (~1Tb) of pyramidal TIF slides.

#### Train

To train the model run

```bash
bash ./train_all.sh
```

On 1 GPU Tesla V100 it will take around 2-3 weeks. If you have more GPUs,
you can [parallelize it](https://www.gnu.org/software/parallel/). Actually,
you do not need to train all ~250 epochs, because the best models on validation
are on ~25 epoch. So you can reduce training to 2-3 days.

To trace models run

```bash
bash ./validate_all.sh
```

Copy generated folders into folder `assets`.

#### Test

Once the model trained run the following command to do inference on test.
Or you can also download the trained models from [yandex disk](https://yadi.sk/d/85zam_5YNLTcQg),
unzip and run

```bash
python main.py
```

### Approach

The main challenge here is to work with extremely high-resolution images (e.g. 150,000 x 85,000 pixels).
One can reduce the resolution by downsampling them to reasonable sizes, such that a deep neural network (DNN)
could work with them but it hurts the performance. On the one hand, we need quite large images with
sufficient information. On the other hand, we should be able to pass them into DNN as an input.

One can notice that most of the image has a monotone background (usually white). The simple
method to reduce the size is to select the tiles/crops based on the number of tissue pixels.
First, divide the image into `N x N` grid. Calculate the sum or mean of pixels' intensity
and then stack top K tiles/crops into one image.

The pipeline is illustrated in the figure below

| C07_B016_S21, page=p4 (`5008x5456` pixels) | grid with `128x128` tiles | stacked top 144 tiles (`1536x1536` pixels) |
|---|---|---|
| ![](./imgs/C07_B016_S21_p4_5008_5456_small.JPG) | ![](./imgs/C07_B016_S21_p4_5008_5456_grid_small.JPG)| ![](./imgs/C07_B016_S21_p4_5008_5456_s128_t144_small.JPG) |

The labels are encoded like

- `0 -> [0, 0, 0]` : benign (normal or subnormal)
- `1 -> [1, 0, 0]` : low malignant potential (low grade squamous intraepithelial lesion)
- `2 -> [1, 1, 0]` : high malignant potential (high grade squamous intraepithelial lesion)
- `3 -> [1, 1, 1]` : invasive cancer (invasive squamous carcinoma)

To predict the final label use the sum with rounding. E. g. we have predictions `[0.6, 0.4, 0.1]`.
The rounded sum is 1. So the label is 1. For `[1.0, 0.3, 0.3]`, label is `2 = round(1.6)`.

The remaining part of the training is the same as for ordinary classification problems.
A convolutional neural network with binary cross-entropy loss with augmentations
on both levels, tiles/crops, and the stacked image is used.

The typical learning curve is

![](./imgs/evolution.png)

#### Highlights

- EfficientNet-B0
- 16x downsampled images (`page` 4)
- 36, 64 and 144 crops with sizes `256x256`, `192x192` and `128x128` respectively
- Binary Cross Entropy Loss
- Batch size 8
- AdamW with learning rate `1e-3` or `3e-4`
- CosineAnealing scheduler
- Augmentations on tile and whole image levels: horizontal and vertical flips, rotate on 90
- Mean predictions of models (8 folds)
