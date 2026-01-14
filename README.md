# 🚗 Camera Occlusion Detection System

**Real-time deep learning–based camera occlusion detection system** for identifying **partial and full camera blockages** from live video streams, designed with a strong focus on **low-latency inference** and **system-level robustness**.

> This project targets safety-critical vision systems where camera failure or obstruction can directly impact decision-making, such as **autonomous driving**, **ADAS**, and **surveillance platforms**.

---

## 🔍 Problem Definition
Vision-based systems often **assume continuous camera availability**.  
In real-world conditions, cameras may be:
- partially covered (dirt, fog, water drops),
- fully blocked (physical obstruction),
- or degraded due to environmental effects.

Most pipelines **do not explicitly detect these failures**, leading to unreliable downstream decisions.

**This project addresses that gap.**

---

## 🧠 Solution Overview
The system performs **continuous, real-time monitoring** of camera input and classifies the camera state as:

- **Normal**
- **Partially Occluded**
- **Fully Occluded**

The design prioritizes:
- **real-time execution**
- **hardware-aware inference**
- **operational reliability over offline accuracy**

---

## ⚙️ High-Level Architecture
```text
Live Camera Feed
        ↓
Frame Preprocessing
        ↓
YOLO-based Occlusion Classifier
        ↓
Occlusion State Prediction
        ↓
Latency & Performance Monitoring
