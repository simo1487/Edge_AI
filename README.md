# Edge_AI

**Edge AI** project: machine learning models are trained on a PC, then run directly on microcontrollers (ESP32), with no server or Internet connection.

## Why Edge AI

Running AI at the edge (on the device itself) instead of in the cloud brings:

- **Low latency**: decisions are made locally, within milliseconds.
- **Privacy**: data (e.g. medical) never leaves the device.
- **Autonomy and low power**: no network required, inexpensive hardware.

The main constraint is size: an ESP32 only has ~320 KB of RAM. The project therefore focuses on building models that are **accurate yet lightweight** (feature selection, tree depth and count, quantization).

## Workflow

1. **Training in Python** (`Mo-dev/Python/`) with scikit-learn: Random Forest, KNN, regression… on several datasets (heart disease, breast cancer, MNIST, etc. in `Mo-dev/data/`).
2. **Analysis**: train/test scores, feature importance, model size reduction.
3. **Export for embedded targets**: conversion of the model to C code with [emlearn](https://github.com/emlearn/emlearn), plus export of the normalization parameters.
4. **Deployment** on ESP32 (PlatformIO / Arduino) to run inference on the board.
