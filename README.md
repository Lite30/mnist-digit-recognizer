# MNIST Digit Recognizer

A simple handwritten digit recognition application with a drawing interface. This project uses TensorFlow/Keras to train a Convolutional Neural Network (CNN) on the MNIST dataset and provides a Tkinter-based GUI for users to draw digits and get predictions.

## Features

- Train a CNN model on the MNIST dataset
- Interactive canvas for drawing digits
- Real-time prediction with confidence visualization
- Simple and intuitive user interface

## File Structure

```
mnist-digit-recognizer/
├── README.md
├── digit_recognizer.py       # Main application file
├── scripts/
│   ├── setup.sh              # Setup script for new environments
│   └── setup_venv.sh         # Setup script for existing virtual environments
├── requirements.txt          # Python dependencies
└── models/                   # Directory for saving trained models (created automatically)
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pillow
- Matplotlib
- Tkinter (usually comes with Python)

## Installation

### Option 1: New Environment Setup

Run the setup script to create a new virtual environment and install all dependencies:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Option 2: Existing Virtual Environment

If you already have a virtual environment, activate it and then run:

```bash
chmod +x scripts/setup_venv.sh
./scripts/setup_venv.sh
```

### Manual Installation

If you prefer to install dependencies manually:

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python digit_recognizer.py
   ```

2. The application will:
   - Load and preprocess the MNIST dataset
   - Train a CNN model (this may take a few minutes)
   - Open a GUI window

3. Draw a digit (0-9) on the black canvas using your mouse
4. Click "Predict" to see the model's prediction
5. Use "Clear" to erase your drawing and try another digit

## How It Works

1. **Model Training**: The application trains a CNN with two convolutional layers followed by max-pooling and dense layers on the MNIST dataset.
2. **Drawing Interface**: Users can draw digits using the mouse on a Tkinter canvas.
3. **Prediction**: When the user clicks "Predict", the drawing is resized to 28×28 pixels (MNIST format) and fed into the model.
4. **Results**: The application shows the predicted digit and confidence levels for each possible digit (0-9).

## Tips for Best Results

- Draw digits that fill most of the canvas
- Use a thick stroke to match the MNIST dataset style
- Center your digits in the canvas
- Try to match the style of handwritten digits from the MNIST dataset (simple, clear digits)

## Troubleshooting

- **TensorFlow installation issues**: If you encounter problems installing TensorFlow, check the [official TensorFlow installation guide](https://www.tensorflow.org/install) for platform-specific instructions.
- **Tkinter not found**: On some Linux distributions, you may need to install Tkinter separately:
  ```
  sudo apt-get install python3-tk  # Ubuntu/Debian
  sudo dnf install python3-tkinter  # Fedora
  ```
- **Slow training**: If training is too slow on your computer, you can reduce the number of epochs in the code (look for `epochs=5` and change it to a smaller number).
