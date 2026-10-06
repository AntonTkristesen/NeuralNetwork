"""
Human Activity Recognition with Keras + Streamlit
=================================================

A neural network learns to guess what a person is doing (walking, sitting, ...)
from smartphone sensor data. The page draws the network itself, so you can watch
it learn and then watch it "think" on data from people it has never seen.

Files:
  app.py     the page: tabs, buttons and what happens when you click them
  data.py    loading the UCI HAR dataset (and what is in it)
  model.py   the neural network: hyperparameters, training with live updates, the forward pass
  charts.py  the network drawing, the accuracy chart and the confusion matrix
  views.py   what the two tabs show
  style.py   colours, fonts and page CSS (the theme itself is in .streamlit/config.toml)

Run (from this folder, so the theme in .streamlit/config.toml is used):
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st

from data import load_dataset
from model import build_model, get_weights, train
from style import PAGE_CSS
from views import show_settings, show_test, show_training

st.set_page_config(page_title="Activity recognition", page_icon=":material/directions_walk:", layout="wide")
st.html(PAGE_CSS)

data = load_dataset()
# Drawn before training starts. Training starts from exactly these random weights (same seed).
untrained = build_model(data.X_train.shape[1], len(data.names))

st.title("Activity recognition")
st.markdown(":gray[A neural network guesses what someone is doing from the motion sensors in their phone.]")
train_tab, test_tab = st.tabs(["Train", "Test on new people"])

with train_tab:
    train_clicked = st.button("Train again" if "result" in st.session_state else "Start training",
                              type="primary", icon=":material/play_arrow:", key="train")
    area = st.empty()  # placeholder that we overwrite after every epoch
    with st.expander("Network and settings"):
        show_settings(data, untrained.count_params())

if train_clicked:
    # st.session_state survives when Streamlit reruns the script (on every click),
    # so the trained model is still there after the next click.
    st.session_state.model, st.session_state.result = train(
        data, redraw=lambda history, cm, weights: show_training(area, history, cm, weights, data))

if "result" in st.session_state:
    r = st.session_state.result
    show_training(area, r["history"], r["test_cm"], get_weights(st.session_state.model), data, result=r)
else:
    show_training(area, [], None, get_weights(untrained), data)

with test_tab:
    if "model" in st.session_state:
        show_test(data, st.session_state.model, st.session_state.result["test_accuracy"])
    else:
        st.info("Train the network first, in the Train tab.")
