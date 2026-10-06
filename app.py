"""
Human Activity Recognition with Keras + Streamlit
=================================================

A neural network learns to guess what a person is doing (walking, sitting, ...)
from smartphone sensor data. The page draws the network itself, so you can watch
it learn and then watch it "think" on data from people it has never seen.

Dataset: UCI "Human Activity Recognition Using Smartphones" (folder "UCI HAR Dataset")
What I found when inspecting the files:
  - 30 persons wore a phone on the waist while doing 6 activities.
  - The sensor signals are cut into windows of 2.56 s (50 % overlap, so a new
    window starts every 1.28 s). For every window, 561 features are computed.
  - train/X_train.txt       7352 rows x 561 numbers (one row = one window), space separated
  - train/y_train.txt       7352 labels, numbers 1-6
  - train/subject_train.txt 7352 person IDs (which person each row belongs to)
  - test/...                the same for 2947 test rows
  - activity_labels.txt     maps 1-6 to names: 1 WALKING ... 6 LAYING
  - features.txt            the names of the 561 features
  - Training = 21 persons, test = 9 OTHER persons (2, 4, 9, 10, 12, 13, 18, 20, 24),
    so the test tells us how well the network works on people it has never seen.
  - The README says the features are already normalized to [-1, 1] -> no scaling needed.
  - "Inertial Signals" holds the raw signals. We don't use them; we use the 561 features.

Run (from this folder, so the theme in .streamlit/config.toml is used):
    pip install -r requirements.txt
    streamlit run app.py
"""

import os

# Keras 3 can run on different "backends" (TensorFlow, JAX or PyTorch).
# We use JAX because TensorFlow had no stable version for Python 3.14 when this
# was written. The Keras code below is identical whichever backend is used.
# This line must come BEFORE "import keras".
os.environ.setdefault("KERAS_BACKEND", "jax")

import math
import sys
import time
from pathlib import Path

import altair as alt
import keras
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "UCI HAR Dataset"

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
# The page is redrawn after every epoch, and in epoch 1 also after these gradient descent steps
EARLY_SNAPSHOTS = (1, 2, 4, 8, 16, 32, 64)

# The drawing shows a sample of the neurons (like the classic "784 inputs" drawings).
# The network itself always uses all of them.
HIDDEN_SHOWN = 24
# The 16 inputs (of 561) the drawing shows: name in features.txt -> plain name
INPUT_FEATURES = {
    "tGravityAcc-mean()-X": "gravity direction x",  # which way the phone is tilted
    "tGravityAcc-mean()-Y": "gravity direction y",
    "tGravityAcc-mean()-Z": "gravity direction z",
    "angle(X,gravityMean)": "tilt angle x",
    "angle(Y,gravityMean)": "tilt angle y",
    "angle(Z,gravityMean)": "tilt angle z",
    "tBodyAccMag-mean()": "movement strength",  # how much the body accelerates
    "tBodyAccMag-std()": "movement variation",
    "tBodyAccJerkMag-mean()": "jerkiness",  # how suddenly the acceleration changes
    "tBodyAcc-std()-X": "movement x",
    "tBodyAcc-std()-Y": "movement y",
    "tBodyAcc-std()-Z": "movement z",
    "tBodyGyroMag-mean()": "rotation strength",  # from the gyroscope
    "tBodyGyro-std()-X": "rotation x",
    "tBodyGyro-std()-Y": "rotation y",
    "tBodyGyro-std()-Z": "rotation z",
}

# Icon + short name for each activity name found in activity_labels.txt
ACTIVITY_INFO = {
    "WALKING": ("🚶", "Walking"),
    "WALKING_UPSTAIRS": ("🚶⬆️", "Upstairs"),
    "WALKING_DOWNSTAIRS": ("🚶⬇️", "Downstairs"),
    "SITTING": ("🪑", "Sitting"),
    "STANDING": ("🧍", "Standing"),
    "LAYING": ("🛏️", "Laying"),
}

# Colours: 3Blue1Brown's manim palette, to match the classic neural network drawings.
# (The theme of the page itself is in .streamlit/config.toml.)
BACKGROUND = "#0E1013"
PANEL = "#1D222A"
CHALK = "#E8E6E1"  # neuron outlines and text
MUTED = "#6B7079"  # training curve, probability bars that are not the guess
BLUE = "#58C4DD"  # a connection that RAISES the next neuron; validation curve
GOLD = "#F0AC5F"  # a connection that LOWERS the next neuron
GREEN = "#83C167"  # the guess is correct
RED = "#FC6255"  # the guess is wrong


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def read_numbers(path):
    """Read a text file with numbers separated by spaces into a numpy array."""
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy()


@st.cache_data(show_spinner="Loading dataset ...")  # read the files only once, not on every click
def load_data():
    # activity_labels.txt has lines like "1 WALKING". Position 0 in the list = label 1, etc.
    labels = pd.read_csv(DATA_DIR / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "name"])
    activity_names = labels.sort_values("id")["name"].tolist()
    # features.txt has lines like "1 tBodyAcc-mean()-X" (shown when you point at an input neuron)
    features = pd.read_csv(DATA_DIR / "features.txt", sep=r"\s+", header=None, names=["id", "name"])
    feature_names = features.sort_values("id")["name"].tolist()

    data = {}
    for part in ["train", "test"]:
        X = read_numbers(DATA_DIR / part / f"X_{part}.txt").astype("float32")  # (rows, 561)
        y = read_numbers(DATA_DIR / part / f"y_{part}.txt").ravel().astype("int32") - 1  # labels 1..6 -> 0..5 (Keras counts from 0)
        person = read_numbers(DATA_DIR / part / f"subject_{part}.txt").ravel()  # person ID per row
        data[part] = (X, y, person)
    return data, activity_names, feature_names


def split_validation(X, y, person, n_val_persons=4):
    """Take the last few training persons out as validation data (used by early stopping).

    We split by PERSON instead of random rows: neighbouring rows overlap by 50 %,
    so a random split would put almost identical rows in both sets. Like this,
    validation measures how well the network works on new people - just like the test.
    """
    val_persons = np.unique(person)[-n_val_persons:]
    is_val = np.isin(person, val_persons)  # True for rows from a validation person
    return X[~is_val], y[~is_val], X[is_val], y[is_val], person[is_val], val_persons


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
def build_model(n_inputs, n_classes):
    """A plain feed-forward network: 561 inputs -> hidden layers -> 6 outputs."""
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


# ---------------------------------------------------------------------------
# CHARTS (made with Altair, which comes with Streamlit: a chart is built from
# layers of marks - lines, circles, text - placed on x/y positions)
# ---------------------------------------------------------------------------
def slot_positions(n):
    """y positions for n drawn neurons, leaving an empty slot in the middle (at y = 0) for the "⋮"."""
    slot = np.arange(n) + (np.arange(n) >= n // 2)
    return slot - n / 2


def network_chart(weights, x, names, feature_names, true=None):
    """Draw the network for one input row x: neuron brightness = activation,
    line = the signal a connection sends (activation of the sending neuron x weight)."""
    acts = forward(weights, x)
    # Which neurons we draw in each layer, and where (y). Inputs: the 16 named features.
    # Hidden layers: an evenly spread sample. Outputs: all 6.
    input_idx = np.array([feature_names.index(f) for f in INPUT_FEATURES])
    hidden_idx = np.linspace(0, N_NEURONS - 1, HIDDEN_SHOWN).round().astype(int)
    out_idx = np.arange(len(names))
    positions = [
        (input_idx, slot_positions(len(input_idx))),
        (hidden_idx, slot_positions(HIDDEN_SHOWN)),
        (hidden_idx, slot_positions(HIDDEN_SHOWN)),
        (out_idx, (out_idx - (len(out_idx) - 1) / 2) * 2.4),
    ]
    titles = [
        f"IN\n{len(x)} numbers from the phone ({len(input_idx)} shown)",
        f"hidden layer 1\n{N_NEURONS} neurons ({HIDDEN_SHOWN} shown)",
        f"hidden layer 2\n{N_NEURONS} neurons ({HIDDEN_SHOWN} shown)",
        "OUT\nhow sure the network is of each activity",
    ]
    # Brightness from 0 (dark) to 1 (white)
    brightness = [
        (acts[0] + 1) / 2,  # inputs lie between -1 and 1
        acts[1] / max(acts[1].max(), 1e-9),  # hidden: relative to the strongest neuron in the layer
        acts[2] / max(acts[2].max(), 1e-9),
        acts[3],  # outputs: the probabilities
    ]

    nodes, texts = [], []
    for layer, (idx, y) in enumerate(positions):
        for i, yy in zip(idx, y):
            name = (INPUT_FEATURES[feature_names[i]] if layer == 0 else names[i] if layer == 3
                    else f"hidden {layer}, neuron {i + 1}")
            nodes.append({"x": layer, "y": yy, "brightness": brightness[layer][i],
                          "neuron": name, "value": round(float(acts[layer][i]), 3)})
            if layer == 0:  # name and value of each input, left of the neuron
                texts.append({"x": -0.06, "y": yy, "text": f"{name}  {acts[0][i]:+.2f}", "size": 12, "align": "right"})
        texts.append({"x": layer, "y": -14.2, "text": titles[layer], "size": 12, "align": "center"})
        if len(idx) < len(acts[layer]):  # not all neurons drawn
            texts.append({"x": layer, "y": 0, "text": "⋮", "size": 18, "align": "center"})

    # One line per connection between the drawn neurons
    edges = []
    for layer, (W, _) in enumerate(weights):
        (src, src_y), (dst, dst_y) = positions[layer], positions[layer + 1]
        signal = acts[layer][src][:, None] * W[np.ix_(src, dst)]  # what each connection passes on
        strength = np.abs(signal) / max(np.abs(signal).max(), 1e-9)
        for a in range(len(src)):
            for c in range(len(dst)):
                edges.append({"x": layer, "y": src_y[a], "x2": layer + 1, "y2": dst_y[c],
                              "strength": strength[a, c], "effect": "raises" if signal[a, c] >= 0 else "lowers"})

    # Probability bars next to the 6 output neurons. The guess is green if correct, red if wrong
    # (plus ✓/✗, so the meaning doesn't depend on colour alone).
    guess = int(np.argmax(acts[3]))
    meters = []
    for i, yy in zip(*positions[3]):
        is_guess = i == guess and true is not None
        color = (GREEN if guess == true else RED) if is_guess else MUTED
        mark = (" ✓" if guess == true else " ✗") if is_guess else ""
        # x positions: name right of the neuron, then the bar (from 3.85, max length 0.75), then the percentage
        meters.append({"y": yy, "name_x": 3.08, "start": 3.85, "end": 3.85 + 0.75 * float(acts[3][i]),
                       "track_end": 4.6, "pct_x": 4.72, "name": names[i], "color": color,
                       "pct": f"{acts[3][i]:.0%}{mark}" + ("   ← true" if i == true else "")})

    x_scale = alt.Scale(domain=[-1.0, 5.3])  # room on the left for the input names
    X = alt.X("x:Q", scale=x_scale, axis=None)
    Y = alt.Y("y:Q", scale=alt.Scale(domain=[-17, 13], reverse=True), axis=None)  # first neuron at the top
    lines = alt.Chart(pd.DataFrame(edges)).mark_rule(strokeWidth=1).encode(
        x=X, y=Y, x2="x2:Q", y2="y2:Q",
        color=alt.Color("effect:N", scale=alt.Scale(domain=["raises", "lowers"], range=[BLUE, GOLD]), legend=None),
        opacity=alt.Opacity("strength:Q", scale=alt.Scale(domain=[0, 1], range=[0, 0.85]), legend=None),
    )
    circles = alt.Chart(pd.DataFrame(nodes)).mark_circle(size=120, stroke=CHALK, strokeWidth=1, opacity=1).encode(
        x=X, y=Y,
        color=alt.Color("brightness:Q", scale=alt.Scale(domain=[0, 1], range=[BACKGROUND, "#FFFFFF"]), legend=None),
        tooltip=["neuron:N", alt.Tooltip("value:Q", title="activation")],
    )
    texts = pd.DataFrame(texts)
    input_names = alt.Chart(texts[texts["align"] == "right"]).mark_text(align="right", color=CHALK).encode(
        x=X, y=Y, text="text:N", size=alt.Size("size:Q", scale=None, legend=None))
    labels = alt.Chart(texts[texts["align"] == "center"]).mark_text(color=MUTED, lineBreak="\n", baseline="bottom").encode(
        x=X, y=Y, text="text:N", size=alt.Size("size:Q", scale=None, legend=None))
    meter = alt.Chart(pd.DataFrame(meters))
    track = meter.mark_rule(strokeWidth=8, strokeCap="round", color=PANEL).encode(
        x=alt.X("start:Q", scale=x_scale, axis=None), x2="track_end:Q", y=Y)
    bar = meter.mark_rule(strokeWidth=8, strokeCap="round").encode(
        x=alt.X("start:Q", scale=x_scale, axis=None), x2="end:Q", y=Y, color=alt.Color("color:N", scale=None))
    out_names = meter.mark_text(align="left", color=CHALK, fontSize=13).encode(
        x=alt.X("name_x:Q", scale=x_scale, axis=None), y=Y, text="name:N")
    pct = meter.mark_text(align="left", color=CHALK, fontSize=12).encode(
        x=alt.X("pct_x:Q", scale=x_scale, axis=None), y=Y, text="pct:N")

    return alt.layer(lines, circles, input_names, labels, track, bar, out_names, pct).properties(height=600)


def confusion_chart(cm, names):
    """Heatmap of the confusion matrix with the count written in every cell."""
    # Altair wants a table with one row per cell: true activity, guess, count
    cells = pd.DataFrame(
        [(names[t], names[g], cm[t, g]) for t in range(len(names)) for g in range(len(names))],
        columns=["true", "guess", "rows"],
    )
    base = alt.Chart(cells).encode(
        x=alt.X("guess:N", sort=names, title="Network's guess", axis=alt.Axis(labelAngle=-40, labelOverlap=False)),
        y=alt.Y("true:N", sort=names, title="True activity"),  # sort=names keeps our order
    )
    squares = base.mark_rect(cornerRadius=2).encode(
        color=alt.Color("rows:Q", scale=alt.Scale(range=[PANEL, BLUE]), legend=None),  # dark = few, bright = many
        tooltip=[alt.Tooltip("true:N", title="True"), alt.Tooltip("guess:N", title="Guess"),
                 alt.Tooltip("rows:Q", title="Rows")],
    )
    # The count in each square: dark text on bright squares, light text on dark squares
    numbers = base.mark_text().encode(
        text="rows:Q",
        color=alt.when(alt.datum.rows > cm.max() / 2).then(alt.value(BACKGROUND)).otherwise(alt.value(CHALK)),
    )
    return (squares + numbers).properties(height=300)


def show_training(area, history, cm, weights, examples, names, feature_names):
    """Draw the training status into 'area' (a placeholder). Replaces what was there before."""
    last = history[-1]
    epoch = last["Epoch"]  # can be a fraction: 0.5 = half of the training rows seen
    with area.container():
        if epoch == 0:
            when = "Before training (random weights)"
        elif epoch < 1:
            when = "Epoch 1, in progress"
        else:
            when = f"Epoch {round(epoch)}"
        st.markdown(f"**{when}** · {last['Steps']:,} gradient descent steps so far · "
                    f"validation accuracy **{last['Validation accuracy']:.1%}** · "
                    f"validation loss {last['Validation loss']:.3f}")
        # Every epoch the drawing shows the next example row: another activity and another person
        x, true, person = examples[int(epoch) % len(examples)]
        activity = names[true].split(" ", 1)[1].lower()
        st.markdown(f"**In:** 2.56 s of phone motion from validation person {person} (the correct answer is "
                    f"**{activity}**) → **Out:** the network's guess")
        st.altair_chart(network_chart(weights, x, names, feature_names, true=true))
        st.caption(
            "The drawing follows one example row and switches to another activity and person every epoch. "
            "Brightness = how strongly a neuron fires. Each line is what a connection passes on "
            "(activation × weight): blue raises the next neuron, gold lowers it."
        )

        df = pd.DataFrame(history).set_index("Epoch")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Accuracy** — share of rows guessed right")
            st.line_chart(df[["Training accuracy", "Validation accuracy"]], color=[MUTED, BLUE], height=260)
        with c2:
            st.markdown("**Loss** — the error gradient descent pushes down")
            st.line_chart(df[["Training loss", "Validation loss"]], color=[MUTED, BLUE], height=260)
        with c3:
            st.markdown("**Confusion matrix** — validation persons")
            st.altair_chart(confusion_chart(cm, names))


def accuracy_and_loss(probs, y):
    """Share of rows guessed right, and the cross-entropy loss: the average of -log(probability
    the network gave the correct answer). (Keras' training loss also adds the small L2 penalty.)"""
    accuracy = np.mean(probs.argmax(axis=1) == y)
    loss = np.mean(-np.log(probs[np.arange(len(y)), y] + 1e-7))
    return float(accuracy), float(loss)


class LiveUpdate(keras.callbacks.Callback):
    """Keras calls these methods during model.fit(): on_train_begin() once at the start,
    on_train_batch_end() after every gradient descent step and on_epoch_end() after every epoch.
    We use them to measure the network and redraw the page."""

    def __init__(self, area, X_train, y_train, X_val, y_val, examples, names, feature_names):
        super().__init__()
        self.area, self.examples, self.names, self.feature_names = area, examples, names, feature_names
        self.X_train, self.y_train, self.X_val, self.y_val = X_train, y_train, X_val, y_val
        self.steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)  # 5867 rows / 64 = 92 steps
        self.history = []  # one entry per snapshot: accuracy and loss on training and validation data
        self.last_cm = None
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
        train_acc, train_loss = accuracy_and_loss(self.model.predict(self.X_train, verbose=0), self.y_train)
        val_probs = self.model.predict(self.X_val, verbose=0)
        val_acc, val_loss = accuracy_and_loss(val_probs, self.y_val)
        self.last_cm = confusion_matrix(self.y_val, val_probs.argmax(axis=1), len(self.names))
        self.history.append({"Epoch": steps / self.steps_per_epoch, "Steps": steps,
                             "Training accuracy": train_acc, "Validation accuracy": val_acc,
                             "Training loss": train_loss, "Validation loss": val_loss})
        show_training(self.area, self.history, self.last_cm, get_weights(self.model),
                      self.examples, self.names, self.feature_names)


# ---------------------------------------------------------------------------
# THE PAGE
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Activity recognition", page_icon="🚶", layout="wide")

data, activity_names, feature_names = load_data()
X_train_all, y_train_all, person_train_all = data["train"]
X_test, y_test, person_test = data["test"]
X_train, y_train, X_val, y_val, person_val, val_persons = split_validation(X_train_all, y_train_all, person_train_all)
names = [" ".join(ACTIVITY_INFO[n]) for n in activity_names]  # e.g. "🚶 Walking"

# The rows the network is drawn for during training, taken from the validation persons.
# The drawing switches to the next one every epoch: another activity AND another person.
examples = []
for k in range(12):
    label, person = k % len(activity_names), val_persons[k % len(val_persons)]
    rows = np.flatnonzero((y_val == label) & (person_val == person))  # this person doing this activity
    row = int(rows[len(rows) // 2])  # a row from the middle of the recording
    examples.append((X_val[row], label, int(person)))

st.title("Recognising activities from phone motion")
st.markdown(
    "30 people wore a smartphone on their waist while they walked, took the stairs, sat, stood and lay down. "
    "Every 1.28 seconds the phone's motion is summarised as 561 numbers. "
    "A neural network has to learn which of the six activities those numbers belong to."
)

# ---------------------------------------------------------------------------
# 1. TRAINING
# ---------------------------------------------------------------------------
st.header("1  Train the network")
n_weights = build_model(X_train.shape[1], len(names)).count_params()
st.markdown(
    f"The network starts with random weights. For every batch of {BATCH_SIZE} rows it guesses the activity, "
    "measures how wrong it was (the **loss**), and **backpropagation** and **gradient descent** nudge all "
    f"{n_weights:,} weights so it is a little less wrong next time. One **epoch** = all "
    f"{len(X_train):,} training rows from 17 people seen once."
)
with st.expander("Network and settings"):
    st.markdown(
        f"- Layers: 561 inputs → {N_LAYERS} hidden layers × {N_NEURONS} neurons (ReLU) → 6 outputs (softmax)\n"
        f"- Learned by gradient descent: {n_weights:,} weights and biases\n"
        f"- Chosen by us (hyperparameters): dropout {DROPOUT}, L2 {L2:g}, learning rate {LEARNING_RATE:g}, "
        f"batch size {BATCH_SIZE}\n"
        f"- Early stopping: stop when validation accuracy hasn't improved for {PATIENCE} epochs "
        "and keep the weights from the best epoch\n"
        f"- Training data: {len(X_train)} rows from 17 persons. Validation: {len(X_val)} rows from persons "
        f"{', '.join(map(str, val_persons))}. Test: {len(X_test)} rows from 9 other persons, used only at the end."
    )
train_clicked = st.button("Start training", type="primary")
area = st.empty()  # placeholder that we overwrite after every epoch

if train_clicked:
    keras.utils.set_random_seed(42)  # same result every time
    model = build_model(X_train.shape[1], len(activity_names))

    # Stop when validation accuracy hasn't improved for PATIENCE epochs in a row,
    # and go back to the weights from the best epoch
    early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True)
    live = LiveUpdate(area, X_train, y_train, X_val, y_val, examples, names, feature_names)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        # Keras needs an epoch count; this one is so large that only early stopping ends training
        epochs=sys.maxsize, batch_size=BATCH_SIZE,
        callbacks=[early_stop, live],
        verbose=0,
    )

    # Final check on the test persons, which the network has never seen
    test_guess = model.predict(X_test, verbose=0).argmax(axis=1)

    # st.session_state survives when Streamlit reruns the script (on every click),
    # so the trained model is still there when we use it below.
    st.session_state.model = model
    st.session_state.result = {
        "history": live.history,
        "val_cm": live.last_cm,
        "best_epoch": early_stop.best_epoch + 1,  # Keras counts epochs from 0
        "test_accuracy": float(np.mean(test_guess == y_test)),
        "test_cm": confusion_matrix(y_test, test_guess, len(names)),
    }

if "result" in st.session_state:
    r = st.session_state.result
    if not train_clicked:  # the page was rerun by another click -> redraw the last training
        show_training(area, r["history"], r["val_cm"], get_weights(st.session_state.model),
                      examples, names, feature_names)
    n_epochs = round(r["history"][-1]["Epoch"])
    st.markdown(
        f"#### Test accuracy: {r['test_accuracy']:.1%}\n"
        f"on the 9 people the network never saw. Training stopped after {n_epochs} epochs "
        f"(no improvement for {PATIENCE}); the weights from epoch "
        f"{r['best_epoch']} are used."
    )
    with st.expander("Confusion matrix on the test persons"):
        st.altair_chart(confusion_chart(r["test_cm"], names))
else:
    area.info("Press **Start training** to watch the network learn.")

# ---------------------------------------------------------------------------
# 2. PLAY WITH THE MODEL
# ---------------------------------------------------------------------------
st.header("2  Try it on people it has never seen")
if "model" not in st.session_state:
    st.info("Train the network first. Then you can play a test person's recording through it here.")
    st.stop()

weights = get_weights(st.session_state.model)
col1, col2, col3 = st.columns([2, 2, 1], vertical_alignment="bottom")
person = col1.selectbox("Test person", np.unique(person_test).tolist())
delay = col2.slider("Seconds per row", 0.0, 1.0, 0.1, step=0.05)
play = col3.button("Play", type="primary", width="stretch")

rows = person_test == person  # True for this person's rows (they are in time order)
X_person, y_person = X_test[rows], y_test[rows]
st.caption(f"Person {person} has {len(X_person)} rows ≈ {len(X_person) * 1.28 / 60:.1f} minutes of recording. "
           "Change any setting to stop the playback.")

if play:
    slot = st.empty()  # placeholder that we overwrite for every row
    n_correct = 0
    for i in range(len(X_person)):
        probs = forward(weights, X_person[i])[-1]  # the 6 output probabilities for this row
        true, guess = int(y_person[i]), int(np.argmax(probs))  # the guess = highest probability
        correct = guess == true
        n_correct += int(correct)

        with slot.container():
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"True activity\n### {names[true]}")
            with c2:
                if correct:
                    st.success(f"Network's guess\n### {names[guess]} ✓ correct")
                else:
                    st.error(f"Network's guess\n### {names[guess]} ✗ wrong")
            c3.markdown(f"Row {i + 1} of {len(X_person)}\n### {n_correct} correct ({n_correct / (i + 1):.0%})")
            st.altair_chart(network_chart(weights, X_person[i], names, feature_names, true=true))
        time.sleep(delay)

    accuracy = n_correct / len(X_person)
    st.markdown(f"#### Accuracy for person {person}: {accuracy:.1%}\n"
                f"All test persons together: {st.session_state.result['test_accuracy']:.1%}.")
