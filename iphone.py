"""Test the network on your own iPhone.

Run, after training the network once in the app (training saves it to model.keras):
    python iphone.py
then open the address it prints in Safari on your iPhone. The iPhone and this computer
must be on the same Wi-Fi.

The iPhone page (iphone.html) sends its raw motion readings here every 1.28 s. This file
translates them into the 561 numbers the network knows (translate.py) and runs the network's
forward pass, the same one as the app's Test tab. The guess goes back to the phone.

Safari only lets a web page use the motion sensors over HTTPS, so this server makes its own
certificate the first time (with openssl). Nobody has signed it, so Safari warns once:
tap "Show Details" -> "visit this website" -> "Visit Website".
"""

import json
import socket
import ssl
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from data import ACTIVITY_NAMES, DATA_DIR, read_names
from model import MODEL_FILE, forward, get_weights  # first: model.py tells Keras to use JAX
import keras
from translate import WINDOW, FS, fit_scaling, gravity_sign, translate

HERE = Path(__file__).parent
PAGE = HERE / "iphone.html"
CERT_DIR = HERE / ".iphone-certificate"
PORT = 8443
MIN_SECONDS = 2 * WINDOW / FS  # 5.12 s: one window, plus as much again for the gravity filter to settle


def local_ip():
    """This computer's address on the Wi-Fi. Connecting a UDP socket sends nothing; it only picks the network."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]


def certificate(ip):
    """A self-made HTTPS certificate for this address (made once per address, valid for a year)."""
    CERT_DIR.mkdir(exist_ok=True)
    cert, key = CERT_DIR / f"{ip}.crt", CERT_DIR / f"{ip}.key"
    if not cert.exists():
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-nodes", "-days", "365",
                        "-keyout", key, "-out", cert, "-subj", f"/CN={ip}",
                        "-addext", f"subjectAltName=IP:{ip}", "-addext", "extendedKeyUsage=serverAuth"],
                       check=True, capture_output=True)
    return cert, key


class Handler(BaseHTTPRequestHandler):
    weights = scaling = names = None  # set in main()

    def do_GET(self):
        if self.path != "/":
            return self.send_error(404)
        self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")

    def do_POST(self):
        steps = {"/flat": self.flat, "/guess": self.guess}
        length = int(self.headers.get("Content-Length", 0))
        if self.path not in steps or length > 2_000_000:
            return self.send_error(404 if self.path not in steps else 413)
        try:
            body = json.loads(self.rfile.read(length))
            readings = np.asarray(body["readings"], dtype=float)
            if readings.ndim != 2 or readings.shape[1] != 7:
                raise ValueError("readings must be rows of 7 numbers")
            answer = steps[self.path](readings, body)
        except (KeyError, TypeError, ValueError) as e:
            answer = {"error": f"{type(e).__name__}: {e}"}
        self.reply(json.dumps(answer).encode(), "application/json")

    def flat(self, readings, body):
        """Step 1: the phone lies flat, screen up. Which sign does this phone give gravity?"""
        sign = gravity_sign(readings)
        if sign is None:
            return {"error": "The phone isn't lying flat. Put it on a table, screen up, and tap Start again."}
        print(f"Gravity sign: {sign:+d}" + (" (flipped, like iOS does)" if sign < 0 else ""))
        return {"sign": sign, "names": self.names}

    def guess(self, readings, body):
        """Step 2: the last few seconds of motion -> 561 numbers -> the network's 6 probabilities."""
        if (readings[-1, 0] - readings[0, 0]) / 1000 < MIN_SECONDS:
            raise ValueError(f"need {MIN_SECONDS:.2f} s of readings")
        x = translate(readings, int(body["sign"]), self.scaling)
        probs = forward(self.weights, x)[-1]  # the same forward pass as the app's Test tab
        guess = int(np.argmax(probs))
        print(f"{self.names[guess]:12s} {probs[guess]:.0%}")
        return {"probabilities": probs.round(4).tolist(), "guess": guess}

    def reply(self, content, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *args):
        pass  # guess() prints one line per guess instead


def main():
    if not MODEL_FILE.exists():
        raise SystemExit(f"No trained network in {MODEL_FILE.name} yet. Train it in the app first: streamlit run app.py")
    print("Loading the network and learning the feature scaling (a few seconds) ...")
    Handler.weights = get_weights(keras.saving.load_model(MODEL_FILE))
    Handler.scaling = fit_scaling()
    Handler.names = [ACTIVITY_NAMES[n] for n in read_names(DATA_DIR / "activity_labels.txt")]

    ip = local_ip()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(*certificate(ip))
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print(f"\nOpen this in Safari on your iPhone (same Wi-Fi):  https://{ip}:{PORT}\nStop with Ctrl+C.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
