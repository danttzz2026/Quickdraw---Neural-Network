"""Tiny Flask server that loads the trained PyTorch model and serves predictions."""
import base64
import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps

from train import QuickDrawCNN

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")

print("Loading model...")
ckpt = torch.load(os.path.join(MODEL_DIR, "quickdraw.pt"), map_location="cpu")
with open(os.path.join(MODEL_DIR, "labels.json")) as f:
    LABELS = json.load(f)
model = QuickDrawCNN(ckpt["num_classes"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Model loaded. {len(LABELS)} classes.")


INK_THRESHOLD = 30   # pixel value above this is considered "ink" (post-invert)
MARGIN_FRAC = 0.10   # padding around the crop, as a fraction of the bbox size


def preprocess(data_url: str) -> torch.Tensor:
    """Decode a canvas image, crop to its ink bounding box, center it on a
    square, and return a (1, 1, 28, 28) tensor that matches the Quick Draw
    training data convention (white-on-black, tight crop, centered)."""
    _header, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGBA")

    # Composite onto white so transparent canvas pixels become white.
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    img = bg.convert("L")

    # Canvas is dark-on-white; training data is white-on-black.
    img = ImageOps.invert(img)
    arr = np.asarray(img, dtype=np.uint8)

    # Find the bounding box of the ink.
    mask = arr > INK_THRESHOLD
    if not mask.any():
        return torch.zeros(1, 1, 28, 28, dtype=torch.float32)

    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    rmin, rmax = np.where(rows_any)[0][[0, -1]]
    cmin, cmax = np.where(cols_any)[0][[0, -1]]
    cropped = arr[rmin : rmax + 1, cmin : cmax + 1]
    h, w = cropped.shape

    # Pad to a square with margin so the doodle sits in roughly the same fraction
    # of the frame as the training bitmaps (~10% border).
    side = max(h, w)
    margin = int(round(side * MARGIN_FRAC))
    target = side + 2 * margin
    canvas = np.zeros((target, target), dtype=np.uint8)
    y_off = (target - h) // 2
    x_off = (target - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = cropped

    resized = Image.fromarray(canvas).resize((28, 28), Image.LANCZOS)
    out = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(out).reshape(1, 1, 28, 28)


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    x = preprocess(data["image"])
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    top_idx = np.argsort(probs)[::-1][:5]
    top = [{"label": LABELS[i], "prob": float(probs[i])} for i in top_idx]
    return jsonify({"predictions": top})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5052, debug=False)
