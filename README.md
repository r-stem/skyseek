# skyseek
A Deep Learning model for detecting anomalies in deep space cosmological observations

## Files

1. **coadd_to_fits.py**: Extracts features from DESI datafiles into .fits files
2. **fits_to_input.py**: Converts .fits files into .npz files for autoecndoer input
3. **skyseek2_autoencoder.py**: Defines the layers of the autoencoeder
4. **skyseek2_train.py**: Trains the autoencoder, including defining architecture and paramaters
5. **skyseek32_classifier.py**: Defines the layers of the classifier
6. **skyseek32_train.py**: Loads the encoder and trains the classifier, including defining architecture and paramaters

