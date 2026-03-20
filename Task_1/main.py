from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam


# Interface
class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input samples."""
        raise NotImplementedError


# Random Forest
class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)

        # Expect (n, 28, 28)
        if X.ndim == 3 and X.shape[1:] == (28, 28):
            X = X.reshape(X.shape[0], -1)
        elif X.ndim == 2 and X.shape[1] == 784:
            pass
        else:
            raise ValueError("Random Forest expects input shape (n, 28, 28).")

        # Normalize pixel values
        if X.max() > 1.0:
            X = X / 255.0

        return X

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_train = self._preprocess(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess(X)
        return self.model.predict(X)


# Feed-Forward Neural Network
class NeuralNetworkMnistClassifier(MnistClassifierInterface):
    def __init__(
        self,
        epochs: int = 5,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3 or X.shape[1:] != (28, 28):
            raise ValueError("Feed-forward NN expects input shape (n, 28, 28).")

        if X.max() > 1.0:
            X = X / 255.0

        return X

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_train = self._preprocess(X_train)
        self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess(X)
        probabilities = self.model.predict(X, verbose=0)
        return np.argmax(probabilities, axis=1)


# Convolutional Neural Network
class CNNMnistClassifier(MnistClassifierInterface):
    def __init__(
        self,
        epochs: int = 5,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax")
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)

        # Expect (n, 28, 28) and convert to (n, 28, 28, 1)
        if X.ndim == 3 and X.shape[1:] == (28, 28):
            X = np.expand_dims(X, axis=-1)
        else:
            raise ValueError("CNN expects input shape (n, 28, 28).")

        if X.max() > 1.0:
            X = X / 255.0

        return X

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_train = self._preprocess(X_train)
        self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess(X)
        probabilities = self.model.predict(X, verbose=0)
        return np.argmax(probabilities, axis=1)


# Wrapper class
class MnistClassifier:
    def __init__(self, algorithm: Literal["rf", "nn", "cnn"]) -> None:
        if algorithm == "rf":
            self.classifier: MnistClassifierInterface = RandomForestMnistClassifier()
        elif algorithm == "nn":
            self.classifier = NeuralNetworkMnistClassifier()
        elif algorithm == "cnn":
            self.classifier = CNNMnistClassifier()
        else:
            raise ValueError("Algorithm must be one of: 'rf', 'nn', 'cnn'.")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.classifier.train(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)


# Example usage with public MNIST dataset
if __name__ == "__main__":
    # Load MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Select algorithm: "rf", "nn", or "cnn"
    model = MnistClassifier(algorithm="cnn")

    # Train
    model.train(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Plot first 12 classified images
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for i in range(12):
        axes[i].imshow(X_test[i], cmap="gray")
        color = "green" if y_pred[i] == y_test[i] else "red"
        axes[i].set_title(
            f"True: {y_test[i]}, Pred: {y_pred[i]}",
            color=color,
            fontweight="bold"
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()