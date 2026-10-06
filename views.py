"""What the two tabs show. Each function draws into the current Streamlit container."""

import time

import numpy as np
import streamlit as st

from charts import accuracy_chart, confusion_chart, network_chart
from data import WINDOW_STEP_SECONDS
from model import BATCH_SIZE, DROPOUT, L2, LEARNING_RATE, N_LAYERS, N_NEURONS, PATIENCE, forward, get_weights

NETWORK_KEY = (":blue[━━] pushes the next neuron up · :orange[━━] pushes it down · "
               "darker circle = more active · only the strongest connections are drawn")


def draw_network(weights, x, true, data):
    """The network drawing for one row, with its colour key underneath."""
    st.altair_chart(network_chart(weights, x, data.names, data.feature_names, true=true))
    st.caption(NETWORK_KEY)


def show_training(area, history, cm, weights, data, result=None):
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
        x, true, person = data.examples[int(epoch) % len(data.examples)]
        with st.container(border=True):
            st.markdown("**Network**")
            st.caption(f"Example from person {person} · actual activity: {data.names[true]}"
                       + ("" if last else " · not trained yet, so the weights are random"))
            draw_network(weights, x, true, data)

        if last:
            left, right = st.columns([2, 3])  # the confusion matrix needs room for 6 labels side by side
            with left.container(border=True):
                st.markdown("**Accuracy**")
                st.altair_chart(accuracy_chart(history))
            with right.container(border=True):
                st.markdown("**Confusion matrix** · " + ("test people" if result else "validation people"))
                st.altair_chart(confusion_chart(cm, data.names))


def show_settings(data, n_weights):
    """The network size, the hyperparameters and how the data is split."""
    st.markdown(
        f"- **Network:** {data.X_train.shape[1]} inputs → {N_LAYERS} hidden layers × {N_NEURONS} neurons → "
        f"{len(data.names)} outputs ({n_weights:,} weights)\n"
        f"- **Settings:** dropout {DROPOUT} · L2 {L2:g} · learning rate {LEARNING_RATE:g} · batch size {BATCH_SIZE}\n"
        f"- **Early stopping:** stops after {PATIENCE} epochs without improvement and keeps the best epoch\n"
        f"- **Training:** {len(data.X_train):,} samples from {data.n_train_persons} people\n"
        f"- **Validation:** {len(data.X_val):,} samples from people {', '.join(map(str, data.val_persons))}\n"
        f"- **Test:** {len(data.X_test):,} samples from {len(np.unique(data.person_test))} other people, "
        "used only at the end\n"
        "- One sample = 2.56 seconds of phone motion, summarised as 561 numbers"
    )


def verdict_html(guess, true, score, names):
    """The row above the drawing in the Test tab: the guess (✓/✗), the actual activity and the score so far."""
    right = guess == true
    return (f'<div class="verdict">'
            f'<div><div class="label">Guess</div><div class="value {"good" if right else "bad"}">'
            f'{names[guess]} {"✓" if right else "✗"}</div></div>'
            f'<div><div class="label">Actual</div><div class="value">{names[true]}</div></div>'
            f'<div><div class="label">Correct so far</div><div class="value">{score}</div></div>'
            f'</div>')


def show_test(data, model, test_accuracy):
    """The Test tab: play a test person's recording through the trained network, one sample at a time."""
    weights = get_weights(model)
    col1, col2, col3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    person = col1.selectbox("Person", np.unique(data.person_test).tolist())
    delay = col2.slider("Seconds per sample", 0.0, 1.0, 0.1, step=0.05)
    play = col3.button("Play", type="primary", icon=":material/play_arrow:", width="stretch")

    rows = data.person_test == person  # True for this person's rows (they are in time order)
    X_person, y_person = data.X_test[rows], data.y_test[rows]
    st.caption(f"{len(X_person)} samples ≈ {len(X_person) * WINDOW_STEP_SECONDS / 60:.1f} minutes of recording. "
               "Changing a setting stops playback.")

    slot = st.empty()  # placeholder that we overwrite for every sample
    n_correct = 0
    for i in range(len(X_person) if play else 1):  # without Play: just show the first sample
        probs = forward(weights, X_person[i])[-1]  # the 6 output probabilities for this sample
        true, guess = int(y_person[i]), int(np.argmax(probs))  # the guess = highest probability
        n_correct += int(guess == true)
        score = f"{n_correct} of {i + 1}" if play else "–"
        with slot.container(border=True):
            st.markdown(verdict_html(guess, true, score, data.names), unsafe_allow_html=True)
            draw_network(weights, X_person[i], true, data)
        if play:
            time.sleep(delay)

    if play:
        st.markdown(f"**Person {person}: {n_correct / len(X_person):.1%} correct.** "
                    f"All test people: {test_accuracy:.1%}.")
