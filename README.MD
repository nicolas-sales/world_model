### World Model for Object Motion Prediction

## Overview

This project implements a world model capable of predicting the future trajectory of objects (cars, pedestrians, etc.) from video data and estimating risk in real time.

The pipeline combines:

Computer Vision (YOLOv8) for detection and tracking
Time-series modeling (Dense / LSTM) for motion prediction
Heuristic reasoning for risk estimation and action recommendation

The system can:

track objects in a video
learn motion patterns
predict future positions
estimate danger levels
generate scene descriptions
suggest driving actions in real time

## Objectives
Extract object trajectories from raw videos
Build a structured dataset with positions and velocities
Train a model to predict future motion
Visualize predicted trajectories
Estimate risk from spatial and temporal features
Generate interpretable outputs (text + actions)

## Pipeline
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
↓
Risk estimation (danger score)
↓
Scene description + action recommendation

## Dataset

# Structure
[video, frame, object_id, type, x, y, conf, vx, vy]

# Preprocessing
Tracking on all frames for ID stability
Frame sampling every dt seconds
Velocity computation using time difference
Per-video normalization:
x = x / width  
y = y / height  
vx = vx / width  
vy = vy / height  

## Temporal Modeling

# Sequence Generation
[t1, t2, t3] → predict t4
Input shape: (sequence_length, 4)
Features: [x, y, vx, vy]

## Models

# Dense Model
Input (3,4)
→ Flatten
→ Dense(64)
→ Dense(64)
→ Dense(4)

# LSTM Model
Input (3,4)
→ LSTM(64)
→ Dense(64)
→ Dense(4)

## Training
Loss: MSE
Metric: MAE
Early stopping
Results
MAE ≈ 0.01 (normalized)
≈ 3.9% average error

The model produces stable and smooth trajectory predictions.

## Inference
Recursive Prediction
[t1, t2, t3] → t4 → t5 → t6 ...

Each prediction is fed back into the model using a sliding window.

## Real-Time System

# Core Features
YOLO tracking on full-resolution frames
Live normalization and velocity computation
Trajectory history tracking
Future trajectory prediction

## Risk Estimation

# Danger Score

The system computes a danger score based on:

distance to ego vehicle (y)
horizontal alignment (x)
relative motion (vy)

This score is used to:

rank objects
select the most critical object

## Future-Aware Risk

# Risk Trend

The system compares:

current danger vs predicted future danger

To estimate:

risk_trend ∈ {stable, increasing, decreasing}

# Stabilization
temporal averaging of predictions
hysteresis to avoid rapid oscillations

## Scene Description

# Generated Output

For the most dangerous object:

A car is very close ahead on the center, approaching fast

## Action Recommendation

# Rule-Based Decision
SAFE
BE CAREFUL
Slow down
BRAKE NOW

Based on the danger score.

## Visualization
Bounding boxes with object IDs
Past trajectory (object color)
Future trajectory (orange)
Danger score display
Main object highlighting

# Overlay Information
Scene description
Recommended action
Risk trend

## Performance Optimizations
Direct inference:
model(x, training=False)
Prediction every 3 frames
Reduced display resolution
Dynamic smoothing based on motion

## Key Insight
Tracking quality > Model complexity

Tracking stability is critical for reliable predictions and risk estimation.

## Current Status
Detection
Tracking
Dataset creation
Model training
Future prediction
Real-time visualization
Risk estimation
Scene description
Action recommendation

## Next Step

# LLM Integration
Convert structured outputs into natural language
Enable reasoning about:
collision risk
object interactions
driving decisions

## Tech Stack
Python
OpenCV
Ultralytics YOLOv8
TensorFlow / Keras
NumPy

## Author

Personal project exploring:

Computer Vision
Time-series prediction
Real-time ML systems
Scene understanding

## Future Vision
Autonomous driving perception modules
Risk-aware navigation systems
AI-assisted driving
Intelligent scene reasoning systems