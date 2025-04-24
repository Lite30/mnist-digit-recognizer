
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class MNISTModel:
    def __init__(self, model_path="models/mnist_model.h5"):
        self.model_path = model_path
        self.model = None
        # Initialize data attributes to avoid AttributeError
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
    def load_data(self):
        print("Loading MNIST dataset...")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        
        # Reshape and normalize the data
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1).astype('float32') / 255
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1).astype('float32') / 255
        
        # Convert labels to categorical
        self.y_train = to_categorical(self.y_train, 10)
        self.y_test = to_categorical(self.y_test, 10)
        print("Dataset loaded and preprocessed.")
        
    def create_model(self):
        print("Creating CNN model...")
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model created and compiled.")
        
    def train_model(self, epochs=5, batch_size=128):
        print(f"Training model for {epochs} epochs...")
        self.model.fit(self.x_train, self.y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_split=0.1,
                      verbose=1)
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save the model
        self.model.save(self.model_path)
        print(f"Model trained and saved to {self.model_path}")
        
    def evaluate_model(self):
        # Only evaluate if we have test data
        if self.x_test is not None and self.y_test is not None:
            print("Evaluating model...")
            score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            print(f"Test loss: {score[0]}")
            print(f"Test accuracy: {score[1]}")
        else:
            print("No test data available for evaluation.")
        
    def load_or_train_model(self, force_train=False):
        # Always load data for evaluation purposes
        self.load_data()
        
        if os.path.exists(self.model_path) and not force_train:
            print(f"Loading existing model from {self.model_path}...")
            self.model = load_model(self.model_path)
            self.evaluate_model()
        else:
            self.create_model()
            self.train_model()
            self.evaluate_model()
            
    def predict(self, image_array):
        if self.model is None:
            self.load_or_train_model()
        
        # Ensure image is properly formatted
        if image_array.shape != (1, 28, 28, 1):
            image_array = image_array.reshape(1, 28, 28, 1)
            
        return self.model.predict(image_array)


class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        
        # Initialize the model
        self.mnist_model = MNISTModel()
        self.mnist_model.load_or_train_model()
        
        # Canvas for drawing
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        
        # Create an image for drawing
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Prediction label
        self.prediction_text = tk.StringVar()
        self.prediction_text.set("Draw a digit (0-9)")
        self.prediction_label = tk.Label(root, textvariable=self.prediction_text, font=("Arial", 16))
        self.prediction_label.grid(row=1, column=0, padx=10, pady=10)
        
        # Confidence bars frame
        self.confidence_frame = tk.Frame(root)
        self.confidence_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10)
        
        # Create confidence bars for each digit
        self.bars = []
        self.bar_labels = []
        for i in range(10):
            label = tk.Label(self.confidence_frame, text=str(i), width=2)
            label.grid(row=i, column=0, padx=5, pady=5)
            
            bar = tk.Canvas(self.confidence_frame, width=200, height=20, bg="white")
            bar.grid(row=i, column=1, padx=5, pady=5)
            bar.create_rectangle(0, 0, 0, 20, fill="blue", tag=f"bar_{i}")
            
            value_label = tk.Label(self.confidence_frame, text="0%", width=5)
            value_label.grid(row=i, column=2, padx=5, pady=5)
            
            self.bars.append(bar)
            self.bar_labels.append(value_label)
        
        # Buttons frame
        button_frame = tk.Frame(root)
        button_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        
        # Clear button
        clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas, width=10)
        clear_button.grid(row=0, column=0, padx=10)
        
        # Predict button
        predict_button = tk.Button(button_frame, text="Predict", command=self.predict_digit, width=10)
        predict_button.grid(row=0, column=1, padx=10)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # For tracking last position
        self.last_x, self.last_y = None, None
    
    def paint(self, event):
        x, y = event.x, event.y
        brush_size = 20
        
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=brush_size, fill="white", capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill=255, width=brush_size)
        else:
            self.canvas.create_oval(x-brush_size/2, y-brush_size/2, x+brush_size/2, y+brush_size/2, fill="white", outline="white")
            self.draw.ellipse([x-brush_size/2, y-brush_size/2, x+brush_size/2, y+brush_size/2], fill=255)
        
        self.last_x, self.last_y = x, y
    
    def on_release(self, event):
        self.last_x, self.last_y = None, None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_text.set("Draw a digit (0-9)")
        
        # Reset confidence bars
        for i in range(10):
            self.bars[i].delete(f"bar_{i}")
            self.bars[i].create_rectangle(0, 0, 0, 20, fill="blue", tag=f"bar_{i}")
            self.bar_labels[i].config(text="0%")
    
    def predict_digit(self):
        # Resize image to 28x28 and convert to numpy array
        img = self.image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255
        
        # Get prediction
        prediction = self.mnist_model.predict(img_array)[0]
        predicted_digit = np.argmax(prediction)
        confidence = prediction[predicted_digit] * 100
        
        # Update prediction text
        self.prediction_text.set(f"Predicted: {predicted_digit} ({confidence:.2f}%)")
        
        # Update confidence bars
        for i in range(10):
            conf_value = prediction[i] * 100
            width = int(conf_value * 2)  # Scale for bar width
            
            self.bars[i].delete(f"bar_{i}")
            self.bars[i].create_rectangle(0, 0, width, 20, fill="blue", tag=f"bar_{i}")
            self.bar_labels[i].config(text=f"{conf_value:.1f}%")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()