"""Translating iPhone sensor data into the 561 numbers the network was trained on.

The network never sees raw sensor data: every input row is 561 features that the dataset's
authors computed from 2.56 s of phone motion (see features_info.txt). To test the network on an
iPhone, we compute those same 561 numbers from the iPhone's own readings.

What is different on the iPhone, and how this file translates it:
  1. Units     Safari on the iPhone reports m/s² and degrees/s. The dataset uses g and radians/s.
  2. Sign      iOS reports gravity with the opposite sign of Android, and the dataset phone was an
               Android (Samsung Galaxy S II). gravity_sign() measures it while the phone lies flat.
  3. Axes      Safari names the gyroscope axes alpha, beta, gamma (= Z, X, Y).
  4. Timing    The iPhone sends ~60 readings per second at uneven times. The dataset has exactly 50.
  5. Features  Noise filter -> gravity/body split -> jerk, magnitude, FFT -> 17 kinds of summary
               numbers, in the order of features.txt.
  6. Scaling   The dataset's features are scaled to [-1, 1], but the scaling numbers were never
               published. We learn them: compute our features from the dataset's own raw signals
               ("Inertial Signals") and fit a straight line per feature onto the dataset's values.

Check how well it works:  python translate.py
"""

import numpy as np
from scipy import signal, stats
from scipy.interpolate import CubicSpline

from data import DATA_DIR, read_numbers

FS = 50  # readings per second in the dataset
WINDOW = 128  # readings in one row (2.56 s)
G = 9.80665  # 1 g in m/s²
AXES = "XYZ"
# bandsEnergy(): FFT bins (1-64) whose energy is added up. Listed in the order of features.txt
BANDS = [(1, 8), (9, 16), (17, 24), (25, 32), (33, 40), (41, 48), (49, 56), (57, 64),
         (1, 16), (17, 32), (33, 48), (49, 64), (1, 24), (25, 48)]


# ---------------------------------------------------------------------------------------------
# 1-4: iPhone readings -> the dataset's units, signs, axes and timing
# ---------------------------------------------------------------------------------------------

def gravity_sign(readings):
    """+1 or -1. Lying flat with the screen up, the dataset's (Android) phone reads +1 g on Z.
    If this phone reads -1 g there, every acceleration has to be flipped. None if it isn't lying flat."""
    z = np.asarray(readings, dtype=float)[:, 3].mean() / G
    return int(np.sign(z)) if abs(abs(z) - 1) < 0.15 else None


def from_iphone(readings, sign):
    """readings: rows of [time (ms), ax, ay, az (m/s², with gravity), alpha, beta, gamma (degrees/s)].
    Returns acceleration (3, n) in g and rotation (3, n) in rad/s, at exactly 50 readings per second."""
    r = np.asarray(readings, dtype=float)
    r = r[np.r_[True, np.diff(r[:, 0]) > 0]]  # drop readings that arrived with the same time stamp
    t = (r[:, 0] - r[0, 0]) / 1000  # seconds since the first reading
    acc = sign * r[:, 1:4] / G
    gyro = np.radians(r[:, [5, 6, 4]])  # beta = X, gamma = Y, alpha = Z
    times = np.arange(0, t[-1], 1 / FS)  # the times the dataset phone would have measured at
    # A smooth curve through the readings. Straight lines between them (np.interp) smooth away fast
    # movements: in a simulated test that cost up to 1 % accuracy.
    return CubicSpline(t, acc)(times).T, CubicSpline(t, gyro)(times).T


# ---------------------------------------------------------------------------------------------
# 5a: the dataset's preprocessing, on a continuous recording (3, n) at 50 Hz
# ---------------------------------------------------------------------------------------------

def low_pass(sig, hz):
    """3rd order Butterworth low-pass filter. Causal (only looks back in time), started at the first reading."""
    b, a = signal.butter(3, hz, fs=FS)
    zi = signal.lfilter_zi(b, a)[None, :] * sig[:, :1]
    return signal.lfilter(b, a, sig, axis=1, zi=zi)[0]


def remove_noise(acc, gyro):
    """Like the dataset: a median filter (3 readings wide; the dataset doesn't say), then a 20 Hz low-pass filter."""
    return low_pass(signal.medfilt(acc, [1, 3]), 20), low_pass(signal.medfilt(gyro, [1, 3]), 20)


def split_gravity(acc):
    """Gravity is the slow part of the acceleration (below 0.3 Hz), body motion is the rest.
    This filter reproduces the dataset's own split (check() measures how closely)."""
    gravity = low_pass(acc, 0.3)
    return acc - gravity, gravity


# ---------------------------------------------------------------------------------------------
# 5b: the 561 features, for windows of shape (rows, 3, 128)
# ---------------------------------------------------------------------------------------------

def entropy(x):
    """Like MATLAB's entropy(), which the dataset's authors used: values are clipped to [0, 1] and rounded
    to 256 levels, then the Shannon entropy (bits) of how often each level occurs. Last axis = readings."""
    levels = np.clip(np.round(x * 255), 0, 255).astype(int).reshape(-1, x.shape[-1])
    counts = np.bincount((np.arange(len(levels))[:, None] * 256 + levels).ravel(), minlength=len(levels) * 256)
    p = counts.reshape(-1, 256) / x.shape[-1]
    return -np.sum(p * np.log2(np.where(p > 0, p, 1)), axis=1).reshape(x.shape[:-1])


def burg(x, order=4):
    """Autoregression coefficients with Burg's method (like MATLAB's arburg). x: (rows, readings)."""
    a = np.ones((len(x), 1))
    f, b = x[:, 1:], x[:, :-1]  # forward and backward prediction errors
    for _ in range(order):
        k = -2 * (f * b).sum(1) / ((f * f).sum(1) + (b * b).sum(1) + 1e-30)
        padded = np.c_[a, np.zeros(len(x))]
        a = padded + k[:, None] * padded[:, ::-1]
        f, b = (f + k[:, None] * b)[:, 1:], (b + k[:, None] * f)[:, :-1]
    return a[:, 1:]


def correlation(a, b):
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + 1e-30)


def angle(a, b):
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
    return np.arccos(np.clip(cos, -1, 1))


def named(name, stat, values):
    """values (rows, axes) -> columns named like features.txt: 'tBodyAcc-mean()-X', or 'tBodyAccMag-mean()'."""
    if values.shape[1] == 1:
        return [(f"{name}-{stat}", values[:, 0])]
    return [(f"{name}-{stat}-{axis}", v) for axis, v in zip(AXES, values.T)]


def summary(name, x):
    """The 9 numbers computed for every signal. x: (rows, axes, readings)."""
    q75, q25 = np.percentile(x, [75, 25], axis=-1, method="hazen")  # "hazen" = how MATLAB computes them
    return (named(name, "mean()", x.mean(-1)) + named(name, "std()", x.std(-1, ddof=1))
            + named(name, "mad()", np.abs(x - x.mean(-1, keepdims=True)).mean(-1))
            + named(name, "max()", x.max(-1)) + named(name, "min()", x.min(-1))
            + [(f"{name}-sma()", np.abs(x).mean(-1).sum(-1))]  # one number for all axes together
            + named(name, "energy()", (x ** 2).mean(-1)) + named(name, "iqr()", q75 - q25)
            + named(name, "entropy()", entropy(x)))


def time_features(name, x):
    """Features of a signal over time: summary + autoregression (+ correlation between the axes)."""
    rows, n_axes, n = x.shape
    ar = burg(x.reshape(-1, n)).reshape(rows, n_axes, 4)
    if n_axes == 1:
        return summary(name, x) + [(f"{name}-arCoeff(){k + 1}", ar[:, 0, k]) for k in range(4)]
    return (summary(name, x)
            + [(f"{name}-arCoeff()-{a},{k + 1}", ar[:, i, k]) for i, a in enumerate(AXES) for k in range(4)]
            + [(f"{name}-correlation()-{AXES[i]},{AXES[j]}", correlation(x[:, i], x[:, j]))
               for i, j in [(0, 1), (0, 2), (1, 2)]])


def frequency_features(name, x):
    """Features of a signal's FFT: how strongly each of the 64 frequencies (0.39 Hz to 25 Hz) is present."""
    spectrum = np.abs(np.fft.fft(x, n=WINDOW, axis=-1))[..., 1:65] * 2 / WINDOW  # without the constant part
    bins = np.arange(1, 65)
    skew, kurt = stats.skew(spectrum, axis=-1), stats.kurtosis(spectrum, axis=-1)
    columns = (summary(name, spectrum) + named(name, "maxInds", spectrum.argmax(-1).astype(float))
               + named(name, "meanFreq()", (spectrum * bins).sum(-1) / (spectrum.sum(-1) + 1e-30)))
    if x.shape[1] == 1:
        return columns + [(f"{name}-skewness()", skew[:, 0]), (f"{name}-kurtosis()", kurt[:, 0])]
    for i, a in enumerate(AXES):
        columns += [(f"{name}-skewness()-{a}", skew[:, i]), (f"{name}-kurtosis()-{a}", kurt[:, i])]
    for i in range(3):
        columns += [(f"{name}-bandsEnergy()-{lo},{hi}", (spectrum[:, i, lo - 1:hi] ** 2).sum(-1)) for lo, hi in BANDS]
    return columns


def jerk(x):
    """How fast a signal changes: the difference between neighbouring readings. Like the dataset,
    not divided by the time between readings, and within the window (127 differences)."""
    return np.diff(x, axis=-1)


def magnitude(x):
    return np.linalg.norm(x, axis=1, keepdims=True)


def features(body, gravity, gyro):
    """All 561 features, unscaled, in the order of features.txt. Each argument: (rows, 3, 128).
    Returns (names, values) with values of shape (rows, 561)."""
    body_jerk, gyro_jerk = jerk(body), jerk(gyro)
    gravity_mean = gravity.mean(-1)
    body_magnitude = time_features("tBodyAccMag", magnitude(body))
    columns = (time_features("tBodyAcc", body) + time_features("tGravityAcc", gravity)
               + time_features("tBodyAccJerk", body_jerk) + time_features("tBodyGyro", gyro)
               + time_features("tBodyGyroJerk", gyro_jerk)
               # The dataset's tGravityAccMag columns are exact copies of tBodyAccMag (a bug in the
               # dataset, easy to check in X_train.txt). The network learned that, so we copy them too.
               + body_magnitude + [(name.replace("tBodyAccMag", "tGravityAccMag"), v) for name, v in body_magnitude]
               + time_features("tBodyAccJerkMag", magnitude(body_jerk)) + time_features("tBodyGyroMag", magnitude(gyro))
               + time_features("tBodyGyroJerkMag", magnitude(gyro_jerk))
               + frequency_features("fBodyAcc", body) + frequency_features("fBodyAccJerk", body_jerk)
               + frequency_features("fBodyGyro", gyro)
               + frequency_features("fBodyAccMag", magnitude(body))
               + frequency_features("fBodyBodyAccJerkMag", magnitude(body_jerk))
               + frequency_features("fBodyBodyGyroMag", magnitude(gyro))
               + frequency_features("fBodyBodyGyroJerkMag", magnitude(gyro_jerk))
               + [("angle(tBodyAccMean,gravity)", angle(body.mean(-1), gravity_mean)),
                  ("angle(tBodyAccJerkMean),gravityMean)", angle(body_jerk.mean(-1), gravity_mean)),
                  ("angle(tBodyGyroMean,gravityMean)", angle(gyro.mean(-1), gravity_mean)),
                  ("angle(tBodyGyroJerkMean,gravityMean)", angle(gyro_jerk.mean(-1), gravity_mean))]
               + [(f"angle({a},gravityMean)", angle(np.eye(3)[i], gravity_mean)) for i, a in enumerate(AXES)])
    names, values = zip(*columns)
    return list(names), np.stack(values, axis=1)


# ---------------------------------------------------------------------------------------------
# 6: scaling to [-1, 1] like the dataset, learned from the dataset itself
# ---------------------------------------------------------------------------------------------

def read_signals(part):
    """The dataset's raw windows for "train" or "test": body, gravity and gyro, each (rows, 3, 128)."""
    folder = DATA_DIR / part / "Inertial Signals"

    def read(kind):
        return np.stack([read_numbers(folder / f"{kind}_{a}_{part}.txt") for a in "xyz"], axis=1)

    total, body = read("total_acc"), read("body_acc")
    return body, total - body, read("body_gyro")


def fit_scaling():
    """A straight line per feature (dataset value = slope * our value + offset), fitted on the training
    rows. The dataset cut its values off at -1 and 1, so rows where that happened are left out of the fit."""
    _, ours = features(*read_signals("train"))
    theirs = read_numbers(DATA_DIR / "train" / "X_train.txt")
    lines = []
    for j in range(ours.shape[1]):
        keep = np.abs(theirs[:, j]) < 0.999
        lines.append(np.polyfit(ours[keep, j], theirs[keep, j], 1))
    return np.array(lines)  # (561, 2): slope, offset


def scale(ours, scaling):
    ours = np.nan_to_num(ours)  # e.g. the skewness of a signal that is exactly 0 (a phone without gyroscope)
    return np.clip(ours * scaling[:, 0] + scaling[:, 1], -1, 1)


def newest_window(acc, gyro, scaling):
    """Noise-free recordings (3, n) at 50 Hz -> the 561 scaled numbers for their newest 2.56 s."""
    body, gravity = split_gravity(acc)
    newest = [s[None, :, -WINDOW:] for s in (body, gravity, gyro)]
    return scale(features(*newest)[1], scaling)[0]


def translate(readings, sign, scaling):
    """The whole translation: iPhone readings (the last few seconds) -> the 561 numbers for the newest 2.56 s."""
    return newest_window(*remove_noise(*from_iphone(readings, sign)), scaling)


# ---------------------------------------------------------------------------------------------
# Checking the translation on the dataset's test people:  python translate.py
# ---------------------------------------------------------------------------------------------

def recordings(total, person):
    """The dataset's windows overlap by half, so a person's consecutive windows can be glued back into
    the continuous recording. Returns the runs: lists of window numbers that belong to one recording."""
    runs = [[0]]
    for i in range(1, len(total)):
        if person[i] == person[i - 1] and np.allclose(total[i, :, :64], total[i - 1, :, 64:], atol=1e-6):
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def glue(windows):
    """Windows (k, 3, 128) of one run -> the continuous recording (3, 64 * (k + 1))."""
    return np.concatenate([*windows[:-1, :, :64], windows[-1]], axis=1)


def as_iphone(acc, gyro):
    """What Safari on an iPhone would report for a dataset recording (g and rad/s at 50 Hz):
    rows of [time (ms), ax, ay, az (m/s², sign flipped), alpha, beta, gamma (degrees/s)] at 60 Hz."""
    t = np.arange(acc.shape[1]) / FS
    times = np.arange(0, t[-1], 1 / 60)
    acc, gyro = CubicSpline(t, -acc.T * G)(times).T, CubicSpline(t, np.degrees(gyro.T))(times).T
    return np.column_stack([times * 1000, *acc, gyro[2], gyro[0], gyro[1]])


def check():
    from data import read_names
    from model import MODEL_FILE, accuracy  # first: model.py tells Keras to use JAX

    import keras

    print("Learning the scaling from the training rows ...")
    scaling = fit_scaling()
    body, gravity, gyro = read_signals("test")
    names, ours = features(body, gravity, gyro)
    ours = scale(ours, scaling)
    theirs = read_numbers(DATA_DIR / "test" / "X_test.txt")
    y = read_numbers(DATA_DIR / "test" / "y_test.txt").ravel() - 1
    person = read_numbers(DATA_DIR / "test" / "subject_test.txt").ravel()
    assert names == read_names(DATA_DIR / "features.txt"), "features are not in the order of features.txt"

    r = np.array([np.corrcoef(ours[:, j], theirs[:, j])[0, 1] for j in range(len(names))])
    print(f"\n1. Our features vs. the dataset's, test people: {(r > 0.99).sum()} of {len(r)} have a correlation "
          f"above 0.99, {(r > 0.9).sum()} above 0.9. Least alike:")
    for j in np.argsort(r)[:5]:
        print(f"     {names[j]:40s} {r[j]:.2f}")

    model = keras.saving.load_model(MODEL_FILE)
    probs = model.predict(theirs, verbose=0)
    print(f"\n2. Network accuracy, test people: {accuracy(probs, y):.1%} on the dataset's features, "
          f"{accuracy(model.predict(ours, verbose=0), y):.1%} on ours")

    # 3. The whole iPhone path, on the recordings glued back together. A phone lying flat reads +1 g on Z
    # in the dataset's convention; as_iphone() flips that like iOS does, and gravity_sign() has to notice.
    sign = gravity_sign(as_iphone(np.tile([[0.0], [0.0], [1.0]], 100), np.zeros((3, 100))))
    total = body + gravity
    rows, phone, gravity_error = [], [], []
    for run in recordings(total, person):
        readings = as_iphone(glue(total[run]), glue(gyro[run]))
        gravity_error.append(np.abs(split_gravity(glue(total[run]))[1] - glue(gravity[run]))[:, 2 * 64:].ravel())
        for k, row in enumerate(run[2:], start=2):  # the phone page also waits 2 windows (5.12 s) before guessing
            end = (64 * k + WINDOW) / FS * 1000  # ms: when this window ends
            recent = readings[(readings[:, 0] <= end) & (readings[:, 0] > end - 12800)]  # the page sends 12.8 s
            rows.append(row)
            # No remove_noise(): the dataset's raw signals were already noise-filtered (a real iPhone's are not)
            phone.append(newest_window(*from_iphone(recent, sign), scaling))
    gravity_error = np.concatenate(gravity_error)
    print(f"\n3. Gravity split vs. the dataset's (after the first 2.56 s): median error {np.median(gravity_error):.5f} g, "
          f"largest {gravity_error.max():.3f} g")
    print(f"\n4. Simulated iPhone (gravity sign measured: {sign:+d}, m/s², degrees/s, 60 Hz), {len(rows)} windows: "
          f"{accuracy(model.predict(np.array(phone), verbose=0), y[rows]):.1%}, "
          f"vs. {accuracy(probs[rows], y[rows]):.1%} on the dataset's features for the same windows")


if __name__ == "__main__":
    check()
