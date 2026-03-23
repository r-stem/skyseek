# Skyseek

A Deep Learning model for detecting anomalies in deep space cosmological observations

## Files

1. **coadd_to_fits.py**: Extracts features from DESI datafiles into .fits files
2. **fits_to_input.py**: Converts .fits files into .npz files for autoecndoer input
3. **skyseek2_autoencoder.py**: Defines the layers of the autoencoder
4. **skyseek2_train.py**: Trains the autoencoder, including defining architecture and paramaters
5. **skyseek32_classifier.py**: Defines the layers of the classifier
6. **skyseek32_train.py**: Loads the encoder and trains the classifier, including defining architecture and paramaters

## Introduction

The Dark Energy Spectroscopic Instrument (DESI) is cosmological observation instrument installed at Kitt Peak National Observatory, Arizona.that observes galaxies and other distant cosmic objects in order to develop a 3D map of the universe. In May 2025, the [first major data release was published](https://data.desi.lbl.gov/doc/releases/dr1/), containing 18 million of those observations.

Manual analysis is untenable for so many objects. To that end, the DESI collaboration has developed an automatic algorithm called ‘redrock’ that uses PCA templates. Redrock is highly reliable (correctly classifying approximately 94% of objects), but struggles with rare and anomalous objects which its templates do not cover. Furthermore, an accuracy of 94% amounts to over a million errors across the dataset, which could hinder research and contaminate the 3D maps.

To that end this model aims to identify objects on which redrock has made a mistake. This will allow for errors to be separated from the rest of the dataset and therefore prevented from contaminating the 3D maps of the universe (redshift is particularly important for this, as it is the primary determinant of the object’s distance in the 3D map).

Ultimately, this project seeks to identify rare and anomalous objects for further study. The current iteration focuses on identifying all redrock errors, which may be due to poor observational quality or other systematic errors.

## Architecture

Skyseek has a hybrid convolutional-attention-MLP architecture designed to interpret spectroscopic data by combining local feature detection with global structural analysis. The encoder portion consists of two convolutional layers followed by two transformer layers. The convolutional layers use kernels of size 6 and 18 to target discrete physical features like emission lines, which typically range from 4 Å to 30 Å in width. These extracted features are then processed by the transformer layers, in order to interpret the global context of and relationships between the detected spectral lines. To finalize the encoding, an attention-pooling layer condenses the output into a 36-length latent vector that captures the most significant information from the input spectra.

In the classification stage, this latent vector is concatenated with 12 redrock metadata values—including redshift, spectral type, and PCA coefficients—resulting in a 48-dimensional input for the classifier. This representation is passed through three shared fully connected layers before splitting into two distinct two-layer MLP heads that independently predict the likelihood of spectral type (S_WRONG) and redshift (Z_WRONG) errors.

![Autoencoder+Classifier Architecture](https://raw.githubusercontent.com/r-stem/skyseek/main/images/32-architecture.svg)

## Training

The model is trained using a combined unsupervised-supervised approach. An autoencoder is trained unsupervised on the vast unlabelled DR1 dataset, then the encoder is attached, weights frozen, to a classifier MLP. This allows for the model to learn to interpret the structure of the data beforehand, so that only classification of these interpretations is trained on the much smaller labeleld dataset.

This approach demonstrated increased performance over a model trained all at once, as shown in Figure 3. F1 score improved by 10.1% for Z_WRONG and 100.8% for S_WRONG.

![A comparison of prediction F1 scores between a model with the encoder trained alongside the classifier (3.0.X), and a model with the encoder trained unsupervised and frozen before classifier training (3.0.1). All other architectural and hyperparamater decisions were kept the same.](images/32-autoencoder-comparison.png)

## Results

| Metric    | Z_Wrong | S_Wrong |
|-----------|---------|---------|
| Threshold | 0.5000  | 0.5975  |
| TPR       | 84.39%  | 62.30%  |
| Precision | 66.85%  | 71.70%  |
| F1        | 0.7461  | 0.6667  |

Skyseek 3.2.2, if run on the DR1 dataset now, could be expected to detect 84.39% of Z_Wrong errors (Ztrue – Z)/(1+Z) > 0.001  and 62.30% of spectral type classification errors. The F1 scores are 0.75 for redshift errors and 0.67 for spectral type errors.

## Next steps

There are still avenues for potentially improving performance, such as data augmentation, using more of the DR1 dataset (only 22% was used due to disk-space requirements), and obtaining a VI set that focuses on the actual main-survey observations (thus avoiding mismatch with the current VI dataset, which tended to have longer exposures).

Furthermore, the model will next be developed to identify rare and anomalous objects through the autoencoder reconstruction error and labelled datasets.
