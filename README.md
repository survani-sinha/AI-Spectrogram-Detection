AI-Spectrogram-Detection

This project estimates the gray level (energy) of each individual packet in a spectrogram image.

It compares two methods:

1. A classical image-processing method
2. A convolutional neural network (CNN)

The dataset contains synthetic spectrogram images with rectangular packets on a near-white background. A darker packet represents a higher gray level and therefore more energy.

Overview

The goal is to estimate the gray level of each packet rather than describing the whole image with one average value.

The classical method works well when packets do not overlap. However, overlapping packets can be detected as one connected region.

The CNN does not need to separate the packets. Instead, it uses the entire image to predict information about the packets and can estimate packets even when they overlap.

Repository Structure

    .
    ├── MethodsForSpectrogramAnomalyDetection.py
    ├── requirements.txt
    ├── data/
    │   ├── images/
    │   └── labels.csv
    └── README.md

Installation

The project uses Python 3.9 or newer.

Install the required packages with:

    pip install -r requirements.txt

The main packages are:

- numpy
- pandas
- matplotlib
- pillow
- scipy
- torch
- scikit-learn

The code was developed to run in Google Colab and uses Google Drive to access the dataset.

A GPU is recommended for CNN training, but the code can also run on a CPU.

Running the Code

The main Python file is:

    MethodsForSpectrogramAnomalyDetection.py

The file contains both methods:

1. Classical image-processing method
2. CNN method

The script:

1. Loads the dataset
2. Evaluates the classical method
3. Trains the CNN
4. Evaluates the CNN
5. Compares the two methods
6. Saves the trained CNN model

The current code expects the dataset to be stored in Google Drive as:

    Take4.zip

The script copies and extracts this file in Google Colab. It then uses:

    /content/Take4/Take4

for the images and labels.

The dataset paths are defined near the beginning of the file:

    IMAGE_DIR = "/content/Take4/Take4"
    LABELS_CSV = "/content/Take4/Take4/labels.csv"

Dataset

The dataset contains 3,000 synthetic 256 × 256 grayscale spectrogram images.

There are three difficulty levels:

- *Sparse:* Packets are spread out and usually do not overlap.
- *Medium:* Packets are placed within 60 pixels of a common center.
- *Dense:* Packets are placed within 20 pixels of a common center and have more overlap.

The images are resized to 128 × 128 before being used by the methods.

The dataset contains:

    Take4/
    ├── images/
    └── labels.csv

Each row in `labels.csv` contains information about one image, including:

- `Filename`
- `AvgGray`
- `PackingDensity`
- `NumBoxes`
- `Gray1`
- `Gray2`
- `Gray3`
- `Gray4`
- `Difficulty`
- `BoxesDrawn`

`Gray1` through `Gray4` are sorted from highest gray level to lowest gray level.

If an image has fewer than four packets, the unused values are `NaN`.

Route 1: Classical Method

The classical method does not require training.

The main steps are:

1. Load the image and resize it to 128 × 128.
2. Estimate the background using the 95th percentile of the pixel values.
3. Create a darkness map by subtracting the image from the estimated background.
4. Smooth the darkness map with a 3 × 3 uniform filter.
5. Calculate a threshold using the median and median absolute deviation (MAD).
6. Clean the resulting mask using morphological operations.
7. Find connected components.
8. Remove components that are too small.
9. Measure the darkness inside each detected packet.
10. Convert the darkness measurement back to a gray-level estimate.
11. Sort the estimated gray levels from highest to lowest.

The method uses a calibration value of `0.3` when converting darkness back to gray level. This value comes from how the synthetic images are generated.

The main limitation is overlapping packets. If two packets overlap, they can become one connected component, causing the classical method to treat them as one packet.

Route 2: CNN

The CNN takes the entire spectrogram image as input and predicts seven values:

- `AvgGray`
- `PackingDensity`
- `NumBoxes`
- `Gray1`
- `Gray2`
- `Gray3`
- `Gray4`

The CNN has six convolutional blocks.

Each block contains:

1. Convolution
2. Batch Normalization
3. ReLU
4. Convolution
5. Batch Normalization
6. ReLU
7. Max Pooling

The convolution kernel sizes are:

    7 × 7
    5 × 5
    5 × 5
    3 × 3
    3 × 3
    3 × 3

The number of channels increases through the network:

    1 → 32 → 64 → 96 → 128 → 128 → 128

The final part of the network is:

1. Adaptive Average Pooling
2. Flatten
3. Dropout
4. Linear layer
5. ReLU
6. Dropout
7. Linear layer

The final linear layer produces seven outputs.

There are 14 weight-bearing layers in total:

- 12 convolutional layers
- 2 linear layers

The model has 1,572,647 trainable parameters.

Loss Function

The CNN uses a masked, weighted Smooth L1 loss.

Smooth L1 is used instead of MSE because it is less sensitive to occasional large errors.

Some images have fewer than four packets. The unused packet outputs are `NaN`, so a mask prevents the model from being penalized for these missing packets.

The output weights are:

    [1.0, 1.0, 0.3, 1.0, 1.0, 1.0, 1.0]

`NumBoxes` receives a smaller weight because it has a larger numerical scale than the other outputs.

Training

The CNN is trained using AdamW.

The main settings are:

- *Learning rate:* `0.001`
- *Weight decay:* `0.0001`
- *Maximum epochs:* `400`
- *Early stopping patience:* `30` epochs
- *Maximum training time:* `90` minutes
- *Training batch size:* `32`
- *Validation and test batch size:* `64`

The learning rate is reduced when the validation loss stops improving.

The targets are standardized using the training data.

Training images are also augmented using:

- Random 90-degree rotations
- Horizontal flips
- Vertical flips

These changes do not affect the packet gray levels or packet count.

Data Split

The dataset is divided into:

- *Training:* 2,099 images
- *Validation:* 451 images
- *Test:* 450 images

The split is stratified by difficulty level.

Training and Validation Loss

The validation loss can sometimes be lower than the training loss.

This happens because dropout is active during training but turned off during validation.

During training, the network is intentionally made harder to use by randomly dropping some neurons. During validation, the full network is used.

Therefore, it is possible for the validation loss to be lower than the training loss.

Using the Code

The script runs the full workflow when executed.

The classical method can also be called on an individual image with:

    per_packet_gray("path/to/image.png")

The CNN and classical method can be compared on one image with:

    analyze("path/to/image.png")

The trained CNN model is saved by the script as:

    /content/specnet_packets.pt

The saved checkpoint contains:

- CNN model weights
- Output normalization values
- Input normalization values
- Output names
- Image size

Outputs

Running the script produces:

- Training and validation loss curves
- Predicted vs. true plots
- R² plots
- A comparison between the two methods
- Route 1 coverage results
- Example predictions
- A saved CNN model checkpoint
