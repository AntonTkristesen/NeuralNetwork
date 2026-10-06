"""The neural network: building it, training it with live updates, and running it by hand."""

import math
import os
import sys
import time

# Keras 3 can run on different "backends" (TensorFlow, JAX or PyTorch).
# We use JAX because TensorFlow had no stable version for Python 3.14 when this
# was written. The Keras code below is identical whichever backend is used.
# This line must come BEFORE "import keras".
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import numpy as np  # noqa: E402

# HYPERPARAMETERS = the settings WE choose. Gradient descent only learns the weights;
# it can't learn how big the network is or how big its steps are.
# These values won a search over 36 combinations, judged on VALIDATION accuracy
# (the test persons were not used to choose them).
N_NEURONS = 256  # neurons per hidden layer
N_LAYERS = 2  # number of hidden layers
DROPOUT = 0.2  # share of neurons switched off at random during training
L2 = 1e-3  # penalty for large weights
LEARNING_RATE = 1e-4  # step size of gradient descent
BATCH_SIZE = 64  # the weights are updated after every 64 rows
PATIENCE = 15  # early stopping: epochs without improvement before training stops (there is no epoch limit)
SEED = 42  # same random start, so the same result every time
# The page is redrawn after every epoch, and in epoch 1 also after these gradient descent steps
EARLY_SNAPSHOTS = (1, 2, 4, 8, 16, 32, 64)


def build_model(n_inputs, n_classes):
    """A plain feed-forward network: 561 inputs -> hidden layers -> 6 outputs.
    It always starts from the same random weights (SEED), so the untrained network
    drawn before training is exactly the one training starts from."""
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential()
    model.add(keras.Input(shape=(n_inputs,)))
    for _ in range(N_LAYERS):
        # L2 regularization adds a small penalty for large weights, so the network
        # learns simpler patterns that work better on new people.
        model.add(keras.layers.Dense(N_NEURONS, activation="relu", kernel_regularizer=keras.regularizers.L2(L2)))
        # Dropout randomly switches off a share of the neurons during training,
        # so the network can't rely on single neurons -> less overfitting.
        model.add(keras.layers.Dropout(DROPOUT))
    # Softmax turns the 6 outputs into probabilities that sum to 1
    model.add(keras.layers.Dense(n_classes, activation="softmax"))

    # The LOSS measures how wrong the network is (sparse_categorical_crossentropy:
    # labels are plain numbers 0-5). After every batch, backpropagation computes how
    # each weight affects the loss (the gradient), and the Adam optimizer (a form of
    # gradient descent) moves every weight a small step in the direction that lowers it.
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def get_weights(model):
    """The learned weights: one (W, b) pair per Dense layer. W[i, j] = weight from neuron i to neuron j."""
    return [layer.get_weights() for layer in model.layers if isinstance(layer, keras.layers.Dense)]


def forward(weights, x):
    """The forward pass written out by hand - the same calculation Keras does for one row.
    (Dropout is only used during training, so it isn't here.)
    Returns the activations of every layer: [inputs, hidden 1, hidden 2, outputs]."""
    activations = [x]
    for i, (W, b) in enumerate(weights):
        z = activations[-1] @ W + b  # for every neuron: weighted sum of the previous layer + bias
        if i < len(weights) - 1:
            a = np.maximum(z, 0)  # ReLU: a negative sum becomes 0 (the neuron doesn't fire)
        else:
            a = np.exp(z - z.max())  # softmax: turns the 6 sums into probabilities that add up to 1
            a = a / a.sum()
        activations.append(a)
    return activations


def confusion_matrix(y_true, y_pred, n_classes):
    """cm[i, j] = number of rows whose TRUE activity is i and which the network GUESSED as j.
    Correct guesses lie on the diagonal; everything else is a mistake."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(probs, y):
    """Share of rows guessed right (the guess = the activity with the highest probability)."""
    return float(np.mean(probs.argmax(axis=1) == y))


class LiveUpdate(keras.callbacks.Callback):
    """Keras calls these methods during model.fit(): on_train_begin() once at the start,
    on_train_batch_end() after every gradient descent step and on_epoch_end() after every epoch.
    We use them to measure the network and call redraw(history, cm, weights) to update the page."""

    def __init__(self, data, redraw):
        super().__init__()
        self.data, self.redraw = data, redraw
        self.steps_per_epoch = math.ceil(len(data.X_train) / BATCH_SIZE)  # 5867 rows / 64 = 92 steps
        self.history = []  # one entry per snapshot: accuracy on training and validation data
        self.step = 0

    def on_train_begin(self, logs=None):
        self.snapshot()  # before any learning: random weights, so no better than chance

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        # The network changes fastest at the start, so in epoch 1 we also look after a few steps
        if self.step in EARLY_SNAPSHOTS:
            self.snapshot()
            time.sleep(0.4)  # short pause, so you can see the first steps

    def on_epoch_end(self, epoch, logs=None):
        self.snapshot()

    def snapshot(self):
        """Measure the network as it is right now and redraw the page."""
        # optimizer.iterations counts how many times gradient descent has updated the weights
        steps = int(self.model.optimizer.iterations)
        val_probs = self.model.predict(self.data.X_val, verbose=0)
        self.history.append({
            "Epoch": steps / self.steps_per_epoch,
            "Training accuracy": accuracy(self.model.predict(self.data.X_train, verbose=0), self.data.y_train),
            "Validation accuracy": accuracy(val_probs, self.data.y_val),
        })
        cm = confusion_matrix(self.data.y_val, val_probs.argmax(axis=1), len(self.data.names))
        self.redraw(self.history, cm, get_weights(self.model))


def train(data, redraw):
    """Train a new network until early stopping ends it, then test it on the test persons.
    redraw(history, cm, weights) is called after every snapshot. Returns (model, result)."""
    model = build_model(data.X_train.shape[1], len(data.names))
    # Stop when validation accuracy hasn't improved for PATIENCE epochs in a row,
    # and go back to the weights from the best epoch
    early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True)
    live = LiveUpdate(data, redraw)
    model.fit(
        data.X_train, data.y_train,
        validation_data=(data.X_val, data.y_val),
        # Keras needs an epoch count; this one is so large that only early stopping ends training
        epochs=sys.maxsize, batch_size=BATCH_SIZE,
        callbacks=[early_stop, live],
        verbose=0,
    )

    # Final check on the test persons, which the network has never seen
    test_guess = model.predict(data.X_test, verbose=0).argmax(axis=1)
    return model, {
        "history": live.history,
        "best_epoch": early_stop.best_epoch + 1,  # Keras counts epochs from 0
        "test_accuracy": float(np.mean(test_guess == data.y_test)),
        "test_cm": confusion_matrix(data.y_test, test_guess, len(data.names)),
    }
