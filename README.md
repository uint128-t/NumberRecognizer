# Number Recognizer

Neural Network to recognize handwritten digits.

Written in Python and does not use any external libraries.

It uses data augmentation so the neural network can better recognize moved or resized digits.

## Configuration

Filenames can be changed in `config.py`, as well as neural network layers, learning rate, and activation functions.

## Training

Download the MNIST dataset (required: `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`) into the current folder.

Then, directly run the `train.py` program. Pypy is recommended, as it runs way faster. The weights will be saved in `weights`

Note: To get a better neural network, training several times is recommended.

## Testing

Testing will not update the weights, and only show accuracy. Requires `t10k-t10k-labels.idx1-ubyte` and `t10k-images.idx3-ubyte`.

## Recognizing

The program only accepts 28x28 8-bit grayscale pictures in raw format, black background. This can be created in GIMP with "Color space" set to "Grayscale".

Then, draw any digit from 0 to 9. Recommended brush is "Paintbrush", size 2px, "Hardness" 80, "Force" 100.

The picture needs to be exported as "Raw data" (`.data`) and saved as `pict.data`.

Then, recognize the digit by running `recognize.py`. It will print the drawing, and print which digit it was recognized as, with a confidence from 0 to 1.

## About

By default, Uses RELM with hidden layers of size 256, 128, 64 and learning rate 0.001.
