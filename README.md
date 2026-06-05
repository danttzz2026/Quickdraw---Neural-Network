# Quick Draw — Doodle Classifier

A small convolutional neural network that recognizes hand-drawn doodles in 15 categories. Built from scratch with PyTorch, served via Flask, with a browser canvas for live predictions.

Draw something with your mouse → top-5 predictions update in real time as you draw.

![architecture](docs/architecture.png)
<sub>(architecture diagram is rendered inline by the web UI — see `web/index.html`)</sub>

---

## What it does

- Trained on Google's [Quick Draw](https://quickdraw.withgoogle.com/data) dataset (15 categories: cat, dog, fish, bird, pizza, banana, apple, car, airplane, bicycle, house, tree, sun, eye, hat)
- **93% validation accuracy** on 15K held-out doodles
- Browser canvas + live prediction bars updating every ~250 ms while you draw
- Server-side preprocessing crops to the bounding box of your ink and recenters before classification — robust to drawings that are off-center, tiny, or in a corner

---

## Architecture

```
input (1×28×28)
    │  Conv 3×3 + ReLU
conv1 (32×28×28)
    │  MaxPool 2×2
pool1 (32×14×14)
    │  Conv 3×3 + ReLU
conv2 (64×14×14)
    │  MaxPool 2×2
pool2 (64×7×7)
    │  Flatten
flatten (3136)
    │  Dense + ReLU + Dropout(0.3)
dense (128)
    │  Dense + Softmax
output (15)
```

~500K trainable parameters.

---

## Stack

- **PyTorch + torchvision** — model definition, training, augmentation
- **NumPy** — data wrangling
- **Flask + Pillow** — inference server + image preprocessing
- **HTML5 Canvas + vanilla JS** — drawing UI

---

## Run it

```bash
pip install -r requirements.txt

python3 download_data.py   # downloads 15 .npy files from Google's bucket (~1.9 GB)
python3 train.py           # trains the CNN, ~5–8 min on Apple MPS / GPU
python3 serve.py           # serves http://localhost:5052
```

Open `http://localhost:5052` and start drawing.

---

## Training details

- **Data**: 30,000 examples per class × 15 classes = 450K training images (out of 100K+ available per class)
- **Augmentation**: random affine (±15° rotation, ±10% translation, 0.85–1.15× scale) applied per-sample at train time
- **Optimizer**: Adam, lr 1e-3, with `ReduceLROnPlateau` (halve when val loss plateaus 2 epochs)
- **Batch size**: 128
- **Epochs**: 12
- **Device**: auto-detects CUDA → MPS → CPU
- **Loss**: cross-entropy

Train loss being notably higher than val loss is expected and healthy — the model is seeing harder, augmented versions of the data than the clean val set.

---

## Inference preprocessing

The browser sends a 280×280 canvas image of your drawing. The server:

1. Composites onto white, converts to grayscale, inverts (training data is white-on-black)
2. Finds the bounding box of inked pixels
3. Crops tightly to that box
4. Pads to a square with ~10% margin (matches Quick Draw's tight-crop convention)
5. Resizes to 28×28 and feeds the model

This is the single biggest reason the model handles tiny doodles in corners and off-center drawings well, despite being trained only on centered samples.

---

## Files

| File | What it does |
|---|---|
| `categories.py` | The 15 class names |
| `download_data.py` | Fetches per-category `.npy` files from `storage.googleapis.com/quickdraw_dataset` (via `curl` — Python's urllib has SSL issues on macOS Python.org installs) |
| `train.py` | Defines `QuickDrawCNN`, the augmented `QuickDrawDataset`, and the training loop |
| `serve.py` | Flask app with `/predict`. Loads the trained model on startup. |
| `web/index.html` | Canvas, prediction bars, architecture diagram |
| `web/app.js` | Mouse/touch drawing handlers; debounced fetch to `/predict` |
| `web/style.css` | Styling |
| `model/quickdraw.pt` | Trained weights (committed for convenience — ~1.6 MB) |
| `model/labels.json` | Class-index → name mapping |

---

## What's left to explore

- Live feature-map preview (show what each conv layer "sees" of your drawing)
- Grad-CAM "why?" heatmap overlaying which strokes mattered most
- Reverse Pictionary game mode (model picks a target, you draw under a timer)
- Expand to 50 / 100 / 345 categories
- Stroke-based model (use Quick Draw's pen sequences instead of bitmaps)
- Public deployment (Hugging Face Spaces)
