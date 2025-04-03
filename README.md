# People Clothing Segmentation Model

This repository contains an implementation of a semantic segmentation model for the "People Clothing Segmentation" dataset. The model can predict 59 clothing classes plus background from images of people.

## Dataset

The model uses the "People Clothing Segmentation" dataset from Kaggle which includes:

- 1000 PNG images (825 x 550 pixels)
- 1000 PNG segmentation masks with 59 clothing classes plus background
- A CSV file listing the classes

## Requirements

The code has been tested with the following dependencies:

- Python 3.8+
- TensorFlow 2.11.0
- NumPy 1.23.5
- pandas 1.5.3
- matplotlib 3.7.1
- scikit-learn 1.2.2
- OpenCV 4.7.0
- tqdm 4.65.0

## Setup

1. Clone this repository
2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Downloading of the "People Clothing Segmentation" dataset happens directly via KaggleHub Python Library

## Model Architecture

The implementation uses a U-Net architecture which is well-suited for semantic segmentation tasks:

- **Encoder**: A series of convolutional and max pooling layers that extract features from the input image
- **Bridge**: Connects the encoder and decoder
- **Decoder**: A series of transposed convolutions (upsampling) and regular convolutions that restore spatial resolution
- **Skip connections**: Connect corresponding encoder and decoder layers to preserve spatial information

## Running the Code

To train and evaluate the model:

```bash
python main.py
```

The script will:

1. Load and preprocess the dataset
2. Split it into training (80%) and validation (20%) sets
3. Build the U-Net model
4. Train the model for 10 epochs with early stopping
5. Evaluate the model on the validation set
6. Save the trained model in both HDF5 and SavedModel formats

## Output

The script generates several outputs:

- `segmentation_epoch_X.png`: Segmentation results at the end of epoch X
- `training_history.png`: Plots of loss and IoU metrics during training
- `validation_results.png`: Segmentation results on validation samples
- `best_model.h5`: The best model weights based on validation IoU
- `segmentation_model.h5`: The final model in HDF5 format
- `saved_model/`: The final model in TensorFlow SavedModel format

## Performance

The model is evaluated using:

- Categorical cross-entropy loss
- Accuracy
- Mean Intersection over Union (IoU)

## Customization

The script includes several hyperparameters that can be adjusted:

- `BATCH_SIZE`: Number of samples per batch (default: 4)
- `EPOCHS`: Maximum number of training epochs (default: 10)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)
- `VALIDATION_SPLIT`: Fraction of data used for validation (default: 0.2)
