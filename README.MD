# World Model for Object Motion Prediction

## Overview

This project implements a **world model** capable of predicting the **future trajectory of objects** (cars, pedestrians, etc.) from video data.

The pipeline combines:

* **Computer Vision (YOLOv8)** for detection and tracking
* **Time-series modeling (LSTM / Dense)** for motion prediction

The result is a system that can:

* track objects in a video
* learn their motion patterns
* predict their future positions in real time

---

## Objectives

* Extract object trajectories from raw videos
* Build a structured dataset with positions and velocities
* Train a model to predict future motion
* Visualize predicted trajectories on video

---

## Pipeline

```
Video
↓
YOLOv8 (Detection + Tracking)
↓
Object positions (x, y)
↓
Velocity computation (vx, vy)
↓
Dataset creation
↓
Sequence generation (sliding window)
↓
World Model (Dense / LSTM)
↓
Future trajectory prediction
```

---

## Dataset

Each sample is structured as:

```
[video, frame, object_id, type, x, y, conf, vx, vy]
```

### Key preprocessing steps:

* Tracking on **all frames** (for ID stability)
* Frame sampling every **dt seconds**
* Velocity computation using time difference
* Per-video normalization:

  * `x = x / width`
  * `y = y / height`
  * `vx = vx / width`
  * `vy = vy / height`

---

## Temporal Modeling

Sequences are built using a sliding window:

```
[t1, t2, t3] → predict t4
```

* Input shape: `(sequence_length, 4)`
* Features: `[x, y, vx, vy]`

---

## Models

### Baseline (Dense)

```python
Input (3,4)
→ Flatten
→ Dense(64)
→ Dense(64)
→ Dense(4)
```

### Improved Model (LSTM)

```python
Input (3,4)
→ LSTM(64)
→ Dense(64)
→ Dense(4)
```

LSTM improves temporal understanding and trajectory smoothness.

---

## Training

* Loss: **MSE**
* Metric: **MAE**
* Early stopping used to prevent overfitting

### Results:

* MAE ≈ **0.01 (normalized)**
* ≈ **Average error of 3.9% 

✔ Stable convergence
✔ No overfitting
✔ Smooth trajectory predictions

---

## Inference

The model predicts multiple future steps using a recursive approach:

```
[t1, t2, t3] → t4 → t5 → t6 ...
```

Each prediction is fed back into the model (sliding window).

---

## Real-Time Integration

Features implemented:

* YOLO tracking on full-resolution frames
* Live normalization
* Velocity computation
* Trajectory history tracking
* Future prediction overlay

### Visualization:

* Past trajectory → object color
* Future trajectory → red

---

## Performance Optimizations

* Replaced `model.predict()` with direct model call:

  ```python
  model(x, training=False)
  ```
* Prediction every **2 frames**
* Lightweight display resolution (while keeping full-res detection)

---

## Example Output

* Smooth predicted trajectories
* Stable motion estimation
* Real-time visualization

---

## Possible Improvements

* Increase sequence length (better temporal context)
* Multi-object interaction modeling
* Collision prediction
* Trajectory smoothing (Kalman filter)
* Graph Neural Networks for scene understanding

---

## Key Insight

```
Tracking quality > Model complexity
```

The performance of the system strongly depends on the quality of object tracking.

---

## Current Status

✔ Detection
✔ Tracking
✔ Dataset creation
✔ Model training
✔ Future prediction
✔ Real-time visualization


---

## Tech Stack

* Python
* OpenCV
* Ultralytics YOLOv8
* TensorFlow / Keras
* NumPy / Pandas

---

## Author

Developed as a personal project exploring:

* Computer Vision
* Time-series prediction
* Real-time ML systems

---

## Future Vision

This project can evolve into:

* Autonomous driving simulation tools
* Behavior prediction systems
* Smart surveillance analytics

---

*A step toward learning how machines understand and predict the physical world.*
