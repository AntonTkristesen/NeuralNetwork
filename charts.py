"""The charts, made with Altair (comes with Streamlit): a chart is built from
layers of marks - lines, circles, text - placed on x/y positions."""

import altair as alt
import numpy as np
import pandas as pd

from model import N_NEURONS, forward
from style import BLUE, DEEP_BLUE, FONT, GRAY, GREEN, HAIRLINE, INK, MONO, MUTED, NEURON_OFF, ORANGE, RED, TRACK

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


def network_chart(weights, x, names, feature_names, true):
    """Draw the network for one input row x (whose actual activity is 'true').
    A neuron's colour shows how strongly it fires; a curve shows the signal a connection
    passes on (activation of the sending neuron x weight). Next to the outputs: how sure
    the network is of each activity."""
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
