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
HIDDEN_SHOWN = 16
# Only the strongest connections between the drawn neurons are drawn, so the picture stays readable
EDGES_SHOWN = 40  # per pair of neighbouring layers
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

# Short name for each activity name found in activity_labels.txt
ACTIVITY_NAMES = {
    "WALKING": "Walking",
    "WALKING_UPSTAIRS": "Upstairs",
    "WALKING_DOWNSTAIRS": "Downstairs",
    "SITTING": "Sitting",
    "STANDING": "Standing",
    "LAYING": "Lying",
}

# Colours. The page is black on white (theme in .streamlit/config.toml);
# colour is only used for data, so wherever it appears it means something.
INK = "#15171C"  # text
MUTED = "#69707D"  # secondary text and axis labels
HAIRLINE = "#E3E7EE"  # gridlines
TRACK = "#EEF1F5"  # empty probability bars, the lightest cells of the confusion matrix
NEURON_OFF = "#DDE3EB"  # a neuron that doesn't fire
BLUE = "#2A78D6"  # a connection that pushes the next neuron UP; the validation curve
DEEP_BLUE = "#1C5CAB"  # a neuron firing at full strength
ORANGE = "#EB6834"  # a connection that pushes the next neuron DOWN
GRAY = "#B4BAC4"  # the training curve; probabilities that aren't the guess
GREEN = "#0CA30C"  # the guess is right
GREEN_TEXT = "#0A7D0A"  # the same green, dark enough for text
RED = "#D03B3B"  # the guess is wrong
FONT = "Geist, system-ui, sans-serif"  # loaded by the theme
MONO = "'Geist Mono', ui-monospace, monospace"

# Layout tweaks the theme can't do: a narrower page and the guess/actual row in the Test tab
PAGE_CSS = f"""<style>
[data-testid="stMainBlockContainer"] {{ max-width: 1100px; padding-top: 3rem; padding-bottom: 5rem; }}
h1 {{ letter-spacing: -0.03em; }}
.verdict {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin-bottom: 0.5rem; }}
.verdict .label {{ font-size: 0.875rem; color: {MUTED}; }}
.verdict .value {{ font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; }}
.verdict .good {{ color: {GREEN_TEXT}; }}
.verdict .bad {{ color: {RED}; }}
@media (max-width: 640px) {{ .verdict {{ grid-template-columns: 1fr; gap: 0.5rem; }} }}
</style>"""

NETWORK_KEY = (":blue[━━] pushes the next neuron up · :orange[━━] pushes it down · "
               "darker circle = more active · only the strongest connections are drawn")


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
def styled(chart):
    """The look all charts share: the page font, quiet grey axes, no frame."""
    return (chart.configure(font=FONT)
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, labelFontSize=11, titleFontSize=11,
                            titleFontWeight="normal", gridColor=HAIRLINE, domainColor=HAIRLINE, tickColor=HAIRLINE)
            .configure_legend(labelColor=MUTED, labelFontSize=11))


def slot_positions(n):
    """y positions for n drawn neurons, leaving an empty slot in the middle (at y = 0) for the "⋮"."""
    slot = np.arange(n) + (np.arange(n) >= n // 2)
    return slot - n / 2


def network_chart(weights, x, names, feature_names, true=None):
    """Draw the network for one input row x. A neuron's colour shows how strongly it fires;
    a curve shows the signal a connection passes on (activation of the sending neuron x weight).
    Next to the outputs: how sure the network is of each activity."""
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
        (out_idx, (out_idx - (len(out_idx) - 1) / 2) * 2.6),
    ]
    titles = [
        ("Input", f"{len(input_idx)} of {len(x)} shown"),
        ("Hidden layer 1", f"{HIDDEN_SHOWN} of {N_NEURONS} shown"),
        ("Hidden layer 2", f"{HIDDEN_SHOWN} of {N_NEURONS} shown"),
        ("Output", ""),
    ]
    # How strongly each neuron fires, from 0 to 1
    fire = [
        (acts[0] + 1) / 2,  # inputs lie between -1 and 1
        acts[1] / max(acts[1].max(), 1e-9),  # hidden: relative to the strongest neuron in the layer
        acts[2] / max(acts[2].max(), 1e-9),
        acts[3],  # outputs: the probabilities
    ]

    nodes, labels = [], []
    for layer, (idx, y) in enumerate(positions):
        for i, yy in zip(idx, y):
            name = (INPUT_FEATURES[feature_names[i]] if layer == 0 else names[i] if layer == 3
                    else f"hidden {layer}, neuron {i + 1}")
            nodes.append({"x": layer, "y": yy, "fire": float(fire[layer][i]), "neuron": name,
                          "activation": round(float(acts[layer][i]), 3)})
        # Layer title and subtitle above the column
        labels.append({"x": layer, "y": -10.8, "text": titles[layer][0], "kind": "title"})
        labels.append({"x": layer, "y": -9.9, "text": titles[layer][1], "kind": "subtitle"})
        if len(idx) < len(acts[layer]):  # not all neurons drawn
            labels.append({"x": layer, "y": 0, "text": "⋮", "kind": "more"})
    nodes, labels = pd.DataFrame(nodes), pd.DataFrame(labels)

    # The strongest connections between the drawn neurons, each drawn as a smooth curve
    edges = []
    for layer, (W, _) in enumerate(weights):
        (src, src_y), (dst, dst_y) = positions[layer], positions[layer + 1]
        signal = acts[layer][src][:, None] * W[np.ix_(src, dst)]  # what each connection passes on
        strength = np.abs(signal) / max(np.abs(signal).max(), 1e-9)
        strongest = np.argsort(strength, axis=None)[::-1][:EDGES_SHOWN]
        for a, c in zip(*np.unravel_index(strongest, strength.shape)):
            if strength[a, c] < 0.02:
                continue
            # Four points: start, two bends half-way, end. Altair smooths them into an S-curve.
            points = [(layer, src_y[a]), (layer + 0.5, src_y[a]), (layer + 0.5, dst_y[c]), (layer + 1, dst_y[c])]
            for order, (px, py) in enumerate(points):
                edges.append({"edge": f"{layer}-{a}-{c}", "order": order, "x": px, "y": py,
                              "strength": float(strength[a, c]),
                              "effect": "up" if signal[a, c] >= 0 else "down"})

    # Probability bars next to the 6 outputs. The guess is green if right, red if wrong
    # (plus ✓/✗, so the meaning doesn't depend on colour alone).
    guess = int(np.argmax(acts[3]))
    meters = []
    for i, yy in zip(*positions[3]):
        p = float(acts[3][i])
        if i != guess:
            color, mark = GRAY, ""
        elif true is None:
            color, mark = BLUE, ""
        else:
            color, mark = (GREEN, " ✓") if guess == true else (RED, " ✗")
        meters.append({"y": yy, "name": names[i], "is_guess": i == guess, "p": p, "start": 3.62,
                       "end": 3.62 + 0.7 * p, "stop": 4.32, "color": color, "pct": f"{p:.0%}{mark}"})
    meters = pd.DataFrame(meters)

    x_scale = alt.Scale(domain=[-0.85, 4.75])  # room on the left for the input names, on the right for the bars
    X = alt.X("x:Q", scale=x_scale, axis=None)
    Y = alt.Y("y:Q", scale=alt.Scale(domain=[-11.5, 8.8], reverse=True), axis=None)  # first neuron at the top
    curves = alt.Chart(pd.DataFrame(edges)).mark_line(interpolate="basis", strokeCap="round").encode(
        x=X, y=Y, detail="edge:N", order="order:Q",
        color=alt.Color("effect:N", scale=alt.Scale(domain=["up", "down"], range=[BLUE, ORANGE]), legend=None),
        opacity=alt.Opacity("strength:Q", scale=alt.Scale(domain=[0, 1], range=[0.1, 0.8]), legend=None),
        strokeWidth=alt.StrokeWidth("strength:Q", scale=alt.Scale(domain=[0, 1], range=[0.5, 2.5]), legend=None),
    )
    circles = alt.Chart(nodes).mark_circle(size=130, opacity=1, stroke="#FFFFFF", strokeWidth=2).encode(
        x=X, y=Y,
        color=alt.Color("fire:Q", scale=alt.Scale(domain=[0, 1], range=[NEURON_OFF, DEEP_BLUE]), legend=None),
        tooltip=[alt.Tooltip("neuron:N", title="Neuron"), alt.Tooltip("activation:Q", title="Activation")],
    )
    input_names = alt.Chart(nodes[nodes["x"] == 0]).mark_text(align="right", dx=-14, fontSize=12, color=MUTED).encode(
        x=X, y=Y, text="neuron:N")
    titles = alt.Chart(labels[labels["kind"] == "title"]).mark_text(fontSize=13, fontWeight=600, color=INK).encode(
        x=X, y=Y, text="text:N")
    subtitles = alt.Chart(labels[labels["kind"] == "subtitle"]).mark_text(fontSize=11, color=MUTED).encode(
        x=X, y=Y, text="text:N")
    more = alt.Chart(labels[labels["kind"] == "more"]).mark_text(fontSize=16, color=GRAY).encode(
        x=X, y=Y, text="text:N")
    meter = alt.Chart(meters)
    track = meter.mark_rule(strokeWidth=6, strokeCap="round", color=TRACK).encode(
        x=alt.X("start:Q", scale=x_scale, axis=None), x2="stop:Q", y=Y)
    bar = meter.transform_filter(alt.datum.p >= 0.01).mark_rule(strokeWidth=6, strokeCap="round").encode(
        x=alt.X("start:Q", scale=x_scale, axis=None), x2="end:Q", y=Y, color=alt.Color("color:N", scale=None))
    out_x = alt.X("out:Q", scale=x_scale, axis=None)
    out_names = meter.transform_calculate(out="3").transform_filter("!datum.is_guess").mark_text(
        align="left", dx=14, fontSize=13, color=MUTED).encode(x=out_x, y=Y, text="name:N")
    guess_name = meter.transform_calculate(out="3").transform_filter("datum.is_guess").mark_text(
        align="left", dx=14, fontSize=13, fontWeight=600, color=INK).encode(x=out_x, y=Y, text="name:N")
    pct = meter.transform_calculate(out="4.4").mark_text(align="left", font=MONO, fontSize=12, color=INK).encode(
        x=out_x, y=Y, text="pct:N")

    chart = alt.layer(curves, circles, input_names, titles, subtitles, more, track, bar, out_names, guess_name, pct)
    return styled(chart.properties(height=470))


def accuracy_chart(history):
    """Accuracy on the training data and on the validation persons, epoch by epoch.
    Point at the chart to read the exact values."""
    wide = pd.DataFrame(history)
    long = wide.melt("Epoch", ["Training accuracy", "Validation accuracy"], var_name="data", value_name="accuracy")
    long["data"] = long["data"].str.removesuffix(" accuracy")
    lines = alt.Chart(long).mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round").encode(
        x=alt.X("Epoch:Q", axis=alt.Axis(grid=False, tickMinStep=1)),
        y=alt.Y("accuracy:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", tickCount=5, domain=False, ticks=False)),
        color=alt.Color("data:N", scale=alt.Scale(domain=["Training", "Validation"], range=[GRAY, BLUE]),
                        legend=alt.Legend(title=None, orient="top", symbolType="stroke", symbolStrokeWidth=2)),
    )
    # An invisible vertical line per epoch that shows up when you point near it, with both values in the tooltip
    hover = alt.selection_point(fields=["Epoch"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    rule = alt.Chart(wide).mark_rule(color=GRAY).encode(
        x="Epoch:Q",
        opacity=alt.when(hover).then(alt.value(1)).otherwise(alt.value(0)),
        tooltip=[alt.Tooltip("Epoch:Q", format=".2~f"),
                 alt.Tooltip("Training accuracy:Q", format=".1%"),
                 alt.Tooltip("Validation accuracy:Q", format=".1%")],
    ).add_params(hover)
    return styled(alt.layer(lines, rule).properties(height=260))


def confusion_chart(cm, names):
    """Confusion matrix: how often each actual activity (row) got each guess (column).
    The diagonal is right; everything else is a mix-up."""
    # Altair wants a table with one row per cell: actual activity, guess, count
    cells = pd.DataFrame(
        [(names[t], names[g], cm[t, g]) for t in range(len(names)) for g in range(len(names))],
        columns=["actual", "guess", "samples"],
    )
    base = alt.Chart(cells).encode(
        x=alt.X("guess:N", sort=names, title="Guess",  # sort=names keeps our order
                axis=alt.Axis(orient="top", labelAngle=0, labelOverlap=False, domain=False, ticks=False,
                              labelPadding=6)),
        y=alt.Y("actual:N", sort=names, title="Actual", axis=alt.Axis(domain=False, ticks=False, labelPadding=8)),
    )
    squares = base.mark_rect(cornerRadius=4, stroke="#FFFFFF", strokeWidth=3).encode(
        color=alt.Color("samples:Q", scale=alt.Scale(domain=[0, int(cm.max())], range=[TRACK, DEEP_BLUE]), legend=None),
        tooltip=[alt.Tooltip("actual:N", title="Actual"), alt.Tooltip("guess:N", title="Guess"),
                 alt.Tooltip("samples:Q", title="Samples")],
    )
    # The count in each square (empty squares stay blank): white on dark squares, black on light ones
    numbers = base.transform_filter(alt.datum.samples > 0).mark_text(font=MONO, fontSize=12).encode(
        text="samples:Q",
        color=alt.when(alt.datum.samples > cm.max() * 0.55).then(alt.value("#FFFFFF")).otherwise(alt.value(INK)),
    )
    return styled((squares + numbers).properties(height=260))


def show_training(area, history, cm, weights, examples, names, feature_names, result=None):
    """Draw the Train tab into 'area' (a placeholder). Replaces what was there before.
    history is empty before training; result is set once training has finished."""
    last = history[-1] if history else None
    epoch = last["Epoch"] if last else 0  # can be a fraction: 0.5 = half of the training rows seen
    with area.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Epoch", f"{epoch:.2f}" if 0 < epoch < 1 else f"{epoch:.0f}", border=True)
        c2.metric("Validation accuracy", f"{last['Validation accuracy']:.1%}" if last else "–", border=True)
        c3.metric("Test accuracy", f"{result['test_accuracy']:.1%}" if result else "–", border=True)
        if result:
            st.caption(f"Stopped after epoch {round(epoch)}: no improvement for {PATIENCE} epochs. "
                       f"Uses the weights from epoch {result['best_epoch']}.")

        # Every epoch the drawing shows the next example: another activity and another person
        x, true, person = examples[int(epoch) % len(examples)]
        with st.container(border=True):
            st.markdown("**Network**")
            st.caption(f"Example from person {person} · actual activity: {names[true]}"
                       + ("" if last else " · not trained yet, so the weights are random"))
            st.altair_chart(network_chart(weights, x, names, feature_names, true=true))
            st.caption(NETWORK_KEY)

        if last:
            left, right = st.columns([2, 3])  # the confusion matrix needs room for 6 labels side by side
            with left.container(border=True):
                st.markdown("**Accuracy**")
                st.altair_chart(accuracy_chart(history))
            with right.container(border=True):
                st.markdown("**Confusion matrix** · " + ("test people" if result else "validation people"))
                st.altair_chart(confusion_chart(cm, names))


def verdict_html(guess, true, score, names):
    """The row above the drawing in the Test tab: the guess (✓/✗), the actual activity and the score so far."""
    right = guess == true
    return (f'<div class="verdict">'
            f'<div><div class="label">Guess</div><div class="value {"good" if right else "bad"}">'
            f'{names[guess]} {"✓" if right else "✗"}</div></div>'
            f'<div><div class="label">Actual</div><div class="value">{names[true]}</div></div>'
            f'<div><div class="label">Correct so far</div><div class="value">{score}</div></div>'
            f'</div>')


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
st.set_page_config(page_title="Activity recognition", page_icon=":material/directions_walk:", layout="wide")
st.html(PAGE_CSS)

data, activity_names, feature_names = load_data()
X_train_all, y_train_all, person_train_all = data["train"]
X_test, y_test, person_test = data["test"]
X_train, y_train, X_val, y_val, person_val, val_persons = split_validation(X_train_all, y_train_all, person_train_all)
names = [ACTIVITY_NAMES[n] for n in activity_names]
test_persons = np.unique(person_test)

# The rows the network is drawn for during training, taken from the validation persons.
# The drawing switches to the next one every epoch: another activity AND another person.
examples = []
for k in range(12):
    label, person = k % len(activity_names), val_persons[k % len(val_persons)]
    rows = np.flatnonzero((y_val == label) & (person_val == person))  # this person doing this activity
    row = int(rows[len(rows) // 2])  # a row from the middle of the recording
    examples.append((X_val[row], label, int(person)))

# The untrained network (random weights) is drawn before training starts. With the same seed,
# training below starts from exactly these weights.
keras.utils.set_random_seed(42)
untrained = build_model(X_train.shape[1], len(names))
n_weights = untrained.count_params()

st.title("Activity recognition")
st.markdown(":gray[A neural network guesses what someone is doing from the motion sensors in their phone.]")
train_tab, test_tab = st.tabs(["Train", "Test on new people"])

# ---------------------------------------------------------------------------
# 1. TRAIN
# ---------------------------------------------------------------------------
with train_tab:
    train_clicked = st.button("Train again" if "result" in st.session_state else "Start training",
                              type="primary", icon=":material/play_arrow:", key="train")
    area = st.empty()  # placeholder that we overwrite after every epoch
    with st.expander("Network and settings"):
        st.markdown(
            f"- **Network:** {X_train.shape[1]} inputs → {N_LAYERS} hidden layers × {N_NEURONS} neurons → "
            f"{len(names)} outputs ({n_weights:,} weights)\n"
            f"- **Settings:** dropout {DROPOUT} · L2 {L2:g} · learning rate {LEARNING_RATE:g} · batch size {BATCH_SIZE}\n"
            f"- **Early stopping:** stops after {PATIENCE} epochs without improvement and keeps the best epoch\n"
            f"- **Training:** {len(X_train):,} samples from {len(np.unique(person_train_all)) - len(val_persons)} people\n"
            f"- **Validation:** {len(X_val):,} samples from people {', '.join(map(str, val_persons))}\n"
            f"- **Test:** {len(X_test):,} samples from {len(test_persons)} other people, used only at the end\n"
            "- One sample = 2.56 seconds of phone motion, summarised as 561 numbers"
        )

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
        "best_epoch": early_stop.best_epoch + 1,  # Keras counts epochs from 0
        "test_accuracy": float(np.mean(test_guess == y_test)),
        "test_cm": confusion_matrix(y_test, test_guess, len(names)),
    }

if "result" in st.session_state:
    r = st.session_state.result
    show_training(area, r["history"], r["test_cm"], get_weights(st.session_state.model),
                  examples, names, feature_names, result=r)
else:
    show_training(area, [], None, get_weights(untrained), examples, names, feature_names)

# ---------------------------------------------------------------------------
# 2. TEST ON PEOPLE THE NETWORK HAS NEVER SEEN
# ---------------------------------------------------------------------------
with test_tab:
    if "model" not in st.session_state:
        st.info("Train the network first, in the Train tab.")
    else:
        weights = get_weights(st.session_state.model)
        col1, col2, col3 = st.columns([2, 2, 1], vertical_alignment="bottom")
        person = col1.selectbox("Person", test_persons.tolist())
        delay = col2.slider("Seconds per sample", 0.0, 1.0, 0.1, step=0.05)
        play = col3.button("Play", type="primary", icon=":material/play_arrow:", width="stretch")

        rows = person_test == person  # True for this person's rows (they are in time order)
        X_person, y_person = X_test[rows], y_test[rows]
        st.caption(f"{len(X_person)} samples ≈ {len(X_person) * 1.28 / 60:.1f} minutes of recording. "
                   "Changing a setting stops playback.")

        slot = st.empty()  # placeholder that we overwrite for every sample
        n_correct = 0
        for i in range(len(X_person) if play else 1):  # without Play: just show the first sample
            probs = forward(weights, X_person[i])[-1]  # the 6 output probabilities for this sample
            true, guess = int(y_person[i]), int(np.argmax(probs))  # the guess = highest probability
            n_correct += int(guess == true)
            score = f"{n_correct} of {i + 1}" if play else "–"
            with slot.container(border=True):
                st.markdown(verdict_html(guess, true, score, names), unsafe_allow_html=True)
                st.altair_chart(network_chart(weights, X_person[i], names, feature_names, true=true))
                st.caption(NETWORK_KEY)
            if play:
                time.sleep(delay)

        if play:
            st.markdown(f"**Person {person}: {n_correct / len(X_person):.1%} correct.** "
                        f"All test people: {st.session_state.result['test_accuracy']:.1%}.")
