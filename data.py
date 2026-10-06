"""Loading the UCI "Human Activity Recognition Using Smartphones" dataset (folder "UCI HAR Dataset").

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
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "UCI HAR Dataset"
WINDOW_STEP_SECONDS = 1.28  # a new 2.56 s window (= one row) starts every 1.28 s
N_VAL_PERSONS = 4  # the last training persons are kept apart as validation data
N_EXAMPLES = 12  # rows drawn during training, one per epoch in turn

# Short name for each activity name found in activity_labels.txt
ACTIVITY_NAMES = {
    "WALKING": "Walking",
    "WALKING_UPSTAIRS": "Upstairs",
    "WALKING_DOWNSTAIRS": "Downstairs",
    "SITTING": "Sitting",
    "STANDING": "Standing",
    "LAYING": "Lying",
}


@dataclass
class Dataset:
    """Everything the app needs from the dataset. Labels are 0-5 (Keras counts from 0)."""
    X_train: np.ndarray  # (rows, 561)
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    person_test: np.ndarray  # person ID of every test row
    val_persons: np.ndarray  # IDs of the validation persons
    n_train_persons: int
    names: list  # short activity name per label, e.g. "Walking"
    feature_names: list  # the 561 names from features.txt (shown when you point at an input neuron)
    examples: list  # (row, label, person) for the rows drawn during training


def read_numbers(path):
    """Read a text file with numbers separated by spaces into a numpy array."""
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy()


def read_names(path):
    """Read a file with lines like "1 WALKING" into a list of names. Position 0 = id 1, etc."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=["id", "name"]).sort_values("id")["name"].tolist()


def read_part(part):
    """X (rows, 561), y (labels 0-5) and the person ID per row for "train" or "test"."""
    X = read_numbers(DATA_DIR / part / f"X_{part}.txt").astype("float32")
    y = read_numbers(DATA_DIR / part / f"y_{part}.txt").ravel().astype("int32") - 1  # labels 1..6 -> 0..5
    person = read_numbers(DATA_DIR / part / f"subject_{part}.txt").ravel()
    return X, y, person


def pick_examples(X_val, y_val, person_val, val_persons, n_labels):
    """The rows drawn during training, from the validation persons. The drawing switches
    to the next one every epoch: another activity AND another person."""
    examples = []
    for k in range(N_EXAMPLES):
        label, person = k % n_labels, val_persons[k % len(val_persons)]
        rows = np.flatnonzero((y_val == label) & (person_val == person))  # this person doing this activity
        row = int(rows[len(rows) // 2])  # a row from the middle of the recording
        examples.append((X_val[row], label, int(person)))
    return examples


@st.cache_data(show_spinner="Loading dataset ...")  # read the files only once, not on every click
def load_dataset():
    names = [ACTIVITY_NAMES[n] for n in read_names(DATA_DIR / "activity_labels.txt")]
    X_all, y_all, person_all = read_part("train")
    X_test, y_test, person_test = read_part("test")

    # Validation = the last few training persons. We split by PERSON instead of random rows:
    # neighbouring rows overlap by 50 %, so a random split would put almost identical rows in
    # both sets. Like this, validation measures how well the network works on new people - just like the test.
    val_persons = np.unique(person_all)[-N_VAL_PERSONS:]
    is_val = np.isin(person_all, val_persons)  # True for rows from a validation person

    return Dataset(
        X_train=X_all[~is_val], y_train=y_all[~is_val],
        X_val=X_all[is_val], y_val=y_all[is_val],
        X_test=X_test, y_test=y_test, person_test=person_test,
        val_persons=val_persons,
        n_train_persons=len(np.unique(person_all)) - len(val_persons),
        names=names,
        feature_names=read_names(DATA_DIR / "features.txt"),
        examples=pick_examples(X_all[is_val], y_all[is_val], person_all[is_val], val_persons, len(names)),
    )
