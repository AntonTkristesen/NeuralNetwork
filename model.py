import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np

# Backpropagation works out, for every weight, which direction lowers the cost (the gradient).
# Gradient descent (Adam) then moves the weights a small step in that direction.
N_NEURONS = 256
N_LAYERS = 2
DROPOUT = 0.2
L2 = 1e-3
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
PATIENCE = 15
SEED = 42
EARLY_SNAPSHOTS = (1, 2, 4, 8, 16, 32, 64)
MODEL_FILE = Path(__file__).parent / "model.keras"  # the trained network, saved by the app, used by iphone.py


def build_model(n_inputs, n_classes):
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential()
    model.add(keras.Input(shape=(n_inputs,)))
    for _ in range(N_LAYERS):
        model.add(keras.layers.Dense(N_NEURONS, activation="relu", kernel_regularizer=keras.regularizers.L2(L2)))
        model.add(keras.layers.Dropout(DROPOUT))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))

    # Training setup: tells Keras HOW to train. Nothing is trained yet; model.fit() uses this later
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),  # attach Adam: the gradient descent that updates the weights
        loss="sparse_categorical_crossentropy",  # the loss: backpropagation computes the gradients of this
        metrics=["accuracy"],  # also measure accuracy during training (early stopping watches it)
    )
    return model


def get_weights(model):
    return [layer.get_weights() for layer in model.layers if isinstance(layer, keras.layers.Dense)]


def forward(weights, x):
    activations = [x]
    for i, (W, b) in enumerate(weights):
        z = activations[-1] @ W + b
        if i < len(weights) - 1:
            a = np.maximum(z, 0)
        else:
            a = np.exp(z - z.max())
            a = a / a.sum()
        activations.append(a)
    return activations


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(probs, y):
    return float(np.mean(probs.argmax(axis=1) == y))


def cross_entropy(probs, y):
    p_right = probs[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(p_right, 1e-7, 1))))


class LiveUpdate(keras.callbacks.Callback):

    def __init__(self, data, redraw):
        super().__init__()
        self.data, self.redraw = data, redraw
        self.steps_per_epoch = math.ceil(len(data.X_train) / BATCH_SIZE)
        self.history = []
        self.step = 0

    def on_train_begin(self, logs=None):
        self.snapshot()

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step in EARLY_SNAPSHOTS:
            self.snapshot()
            time.sleep(0.4)

    def on_epoch_end(self, epoch, logs=None):
        self.snapshot()

    def snapshot(self):
        steps = int(self.model.optimizer.iterations)
        train_probs = self.model.predict(self.data.X_train, verbose=0)
        val_probs = self.model.predict(self.data.X_val, verbose=0)
        penalty = float(sum(self.model.losses))
        self.history.append({
            "Epoch": steps / self.steps_per_epoch,
            "Training accuracy": accuracy(train_probs, self.data.y_train),
            "Validation accuracy": accuracy(val_probs, self.data.y_val),
            "Training cost": cross_entropy(train_probs, self.data.y_train) + penalty,
            "Validation cost": cross_entropy(val_probs, self.data.y_val) + penalty,
        })
        cm = confusion_matrix(self.data.y_val, val_probs.argmax(axis=1), len(self.data.names))
        self.redraw(self.history, cm, get_weights(self.model))


def train(data, redraw):
    model = build_model(data.X_train.shape[1], len(data.names))
    early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True)
    live = LiveUpdate(data, redraw)
    # Training happens here. For every batch of 64 rows Keras does:
    # forward pass -> loss -> backpropagation (gradients) -> Adam updates the weights
    model.fit(
        data.X_train, data.y_train,  # the rows and right answers the loss and gradients are computed from
        validation_data=(data.X_val, data.y_val),  # only measured after each epoch, never learned from
        epochs=sys.maxsize,  # no epoch limit: early stopping ends training
        batch_size=BATCH_SIZE,  # Adam updates the weights after every batch of 64 rows (92 times per epoch)
        callbacks=[early_stop, live],  # not training: early_stop ends it, live redraws the page
        verbose=0,  # don't print progress
    )

    test_guess = model.predict(data.X_test, verbose=0).argmax(axis=1)
    return model, {
        "history": live.history,
        "best_epoch": early_stop.best_epoch + 1,
        "test_accuracy": float(np.mean(test_guess == data.y_test)),
        "test_cm": confusion_matrix(data.y_test, test_guess, len(data.names)),
    }
