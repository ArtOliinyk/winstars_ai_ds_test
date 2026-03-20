# Task 1. Image classification + OOP

An implementation comparing three different machine learning classifiers on the MNIST handwritten digit dataset: Random Forest, Feed-Forward Neural Network, and Convolutional Neural Network.

## Approach

This project implements three classifiers through a unified interface (`MnistClassifierInterface`), allowing fair comparison across different algorithms:

- **Random Forest Classifier**: An ensemble of decision trees for feature-based classification
- **Feed-Forward Neural Network**: A fully connected multi-layer perceptron with non-linear activations
- **Convolutional Neural Network (CNN)**: A deep learning model leveraging spatial structure with convolutional and pooling layers

All classifiers:
- Accept standardized input: `(n, 28, 28)` shaped images
- Output: 1D array of predicted class labels (0-9)
- Normalize pixel values to [0, 1] range
- Are trained on the MNIST dataset

## Architecture

### Class Structure

```
MnistClassifierInterface (ABC)
├── RandomForestMnistClassifier
├── NeuralNetworkMnistClassifier
└── CNNMnistClassifier

MnistClassifier (Wrapper)
```

### RandomForestMnistClassifier
- Flattens 2D images to 1D feature vectors (784 features)
- Uses scikit-learn's RandomForestClassifier with 100 estimators
- Parameters: `n_estimators`, `random_state`

### NeuralNetworkMnistClassifier
- Flattens images and passes through dense layers: 128 → 64 → 10
- Uses ReLU activation for hidden layers, softmax for output
- Adam optimizer with sparse categorical crossentropy loss
- Parameters: `epochs`, `batch_size`, `learning_rate`

### CNNMnistClassifier
- Processes images with spatial structure using convolution
- Architecture: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(64) → Dense(10)
- Uses ReLU activation for hidden layers, softmax for output
- Parameters: `epochs`, `batch_size`, `learning_rate`

## Input/Output Specifications

### Input
All classifiers expect:
- **Shape**: `(num_samples, 28, 28)` - grayscale images
- **Data Type**: numpy ndarray, any numeric type
- **Value Range**: 0-255 (automatically normalized to 0-1)

### Output
All classifiers return:
- **Shape**: `(num_samples,)` - 1D array
- **Data Type**: numpy ndarray of integers
- **Values**: 0-9 (digit predictions)

## Usage Example

```python
from main import MnistClassifier
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Create and train classifier
model = MnistClassifier(algorithm="cnn")  # Options: "rf", "nn", "cnn"
model.train(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Visualize results
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_test[i], cmap="gray")
    color = "green" if y_pred[i] == y_test[i] else "red"
    ax.set_title(f"True: {y_test[i]}, Pred: {y_pred[i]}", color=color)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Installation

1. Ensure Python 3.9+ is installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Script

```bash
python main.py
```

The script will:
1. Load the MNIST training and test datasets
2. Train the selected classifier
3. Make predictions on the test set
4. Display overall accuracy
5. Visualize first 12 test images with true/predicted labels (green=correct, red=incorrect)

## Classifier Selection

Modify the `algorithm` parameter in the usage:
- `"rf"` - Random Forest (fast training, no GPU needed)
- `"nn"` - Feed-Forward Neural Network (medium performance)
- `"cnn"` - Convolutional Neural Network (best accuracy, requires more computation)

## Dependencies

- **numpy**: Numerical operations
- **scikit-learn**: Random Forest implementation
- **tensorflow**: Keras API for neural networks
- **matplotlib**: Data visualization

## Task 1 Demo Notebook

The project includes a Jupyter notebook demonstrating solution behavior and edge-case handling:

- `demo.ipynb` in this folder
- runs all three algorithms on MNIST
- includes edge case tests for invalid input shapes and empty data

## Notes

- Random Forest requires flattening images to 784 features
- Neural networks expect 3D input (28, 28) which is reshaped internally
- CNN preserves spatial structure with channel dimension (28, 28, 1)
- All classifiers normalize pixel values automatically
- GPU support accelerates CNN and neural network training (optional)
