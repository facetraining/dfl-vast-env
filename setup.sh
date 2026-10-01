#!/bin/bash
set -e

echo "=== [1/5] Linking Python to Python3 ==="
ln -sf $(which python3) /usr/local/bin/python

echo "=== [2/5] Installing System Libraries ==="
apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unzip \
    tmux \
    curl \
    git

echo "=== [3/5] Pinning Core Packages ==="
pip install --upgrade pip
pip install --force-reinstall numpy==1.23.5
pip install tensorflow==2.10.1 tensorflow-estimator==2.10.0

echo "=== [4/5] Installing Pinned Environment ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements.txt" --no-deps

echo "=== [5/5] Verification Checklist ==="
python3 -c "
import numpy, tensorflow as tf, PIL, numexpr, ffmpeg
gpus = tf.config.list_physical_devices(\"GPU\")
print(\"NumPy:\", numpy.__version__)
print(\"TF:\", tf.__version__)
print(\"FFmpeg Python Module: OK\")
print(\"GPUs Available:\", len(gpus))
assert len(gpus) > 0, \"No GPU detected!\"
print(\"--> ALL CHECKS PASSED. Ready to train/merge.\")
"
