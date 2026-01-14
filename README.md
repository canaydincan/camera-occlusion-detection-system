---

## 🏷️ Occlusion Classes & Dataset Design

The system is formulated as a **multi-class camera reliability classification problem**, focusing on identifying the **source of visual degradation** rather than treating occlusion as a single binary event.

This design choice enables **more informative diagnostics** and improves reliability in safety-critical vision systems.

---

## 📚 Defined Occlusion Classes

- **🟢 Normal**  
  Represents a clear and unobstructed camera view under nominal operating conditions.  
  This class serves as the baseline reference for system behavior.

- **🧍 Human Occlusion**  
  Partial or full occlusion caused by **direct human interaction**, such as a hand covering the camera lens.  
  Common in tampering, testing, or accidental misuse scenarios.

- **🧱 Object Occlusion**  
  Camera blockage caused by **physical objects** (e.g., tools, covers, debris).  
  Includes both intentional and unintentional obstruction cases.

- **🌧️ Weather Occlusion**  
  Visibility degradation due to **environmental factors** such as rain, fog, snow, or mud.  
  Particularly critical for autonomous driving and outdoor perception systems.

- **⚙️ Sensor Failure**  
  Non-visual degradation including **camera blackout, signal loss, or hardware malfunction**.  
  Treated as a separate class to distinguish sensor-level faults from physical occlusions.

---

## 📊 Dataset Distribution

The dataset reflects **real-world class imbalance**, which is intentionally preserved to improve robustness during deployment:

- **Normal:** 439 samples  
- **Weather Occlusion:** 188 samples  
- **Sensor Failure:** 59 samples  
- **Human Occlusion:** 45 samples  
- **Object Occlusion:** 44 samples  

No artificial rebalancing was applied, allowing the model to learn under **realistic operating conditions**.

---

## 🧠 Design Rationale

Most vision-based systems implicitly assume continuous camera availability and only ask:

> *“Is the camera blocked?”*

This project instead focuses on a more operationally meaningful question:

> **“Why is the camera unreliable?”**

By explicitly modeling **human**, **object**, **weather**, and **sensor-level** failure modes, the system supports more reliable downstream decision-making and safer system behavior.

---

## 🛠️ Annotation Strategy

- **Annotation Tool:** Roboflow  
- **Labeling Scheme:** Single-label, mutually exclusive classes  
- **Class Policy:** Annotation classes locked to prevent label drift  

This ensures dataset consistency across iterations and contributors.

---

## 🧠 Model & Dataset

The system is built as a **multi-class deep learning–based camera reliability classifier**, trained to identify not only whether a camera is occluded, but **how and why** its visual input is degraded.

The model operates directly on raw camera frames and learns discriminative patterns associated with:
- physical obstruction,
- environmental interference,
- and sensor-level failures.

The dataset was custom-designed to reflect **real-world operating conditions**, including naturally occurring class imbalance.  
Rather than enforcing artificial balance, the dataset preserves realistic failure frequencies to improve deployment robustness.

---

## ⚡ Performance & Hardware

The system is designed for **real-time inference**, with hardware-aware optimizations to support different execution backends.

Performance evaluations were conducted on multiple platforms:

- **Apple Silicon (MPS Backend):**  
  Provided stable inference latency and consistent real-time performance, making it suitable for live demonstrations and continuous monitoring.

- **NVIDIA RTX 3060 Ti (CUDA Backend):**  
  Achieved higher throughput, but exhibited occasional peak latency spikes due to GPU scheduling and memory transfer overhead.

These results highlight an important **engineering trade-off** between peak performance and latency stability across heterogeneous hardware.

---

## 📊 Key Metrics

The system was evaluated using metrics relevant to **real-time, safety-critical vision systems**:

- **Top-1 Classification Accuracy:** > 90%  
- **Real-Time Inference:** Enabled  
- **Latency Monitoring:** Continuous (including peak latency observation)  
- **FPS Tracking:** Live during execution  
- **Cross-Platform Support:** macOS (MPS) and NVIDIA CUDA GPUs  

Performance metrics are displayed during runtime to provide immediate feedback on system health.

---

## 🧪 Use Cases

This system is applicable to any scenario where **silent camera failure can lead to unsafe decisions**, including:

- Autonomous driving perception pipelines  
- Advanced Driver Assistance Systems (ADAS)  
- Outdoor surveillance and monitoring  
- Robotics and mobile platforms relying on visual input  
- Camera health monitoring for long-running deployments  

The model can be integrated as a **pre-check or watchdog module** before downstream perception tasks.

---

## ⚠️ Known Limitations

- Model performance is influenced by hardware backend and driver-level scheduling behavior  
- Extreme or rare environmental conditions may require additional domain-specific data  
- The system identifies the **type of failure**, not the exact physical root cause  

These limitations are explicitly documented as part of a **transparent engineering design process**.

---

## ⏱️ Latency & Real-Time Constraints

Since the system targets **safety-critical, real-time vision applications**, inference latency is treated as a **primary design constraint**, not a secondary optimization metric.

The system continuously measures and reports **end-to-end inference latency in milliseconds (ms)** during runtime.

Latency evaluation focuses on:

- **Per-frame inference time (ms)**  
- **Peak latency behavior under load**  
- **Latency stability across different hardware backends**

Rather than reporting only average latency values, the system explicitly monitors **peak latency**, as worst-case delays are more relevant for real-world deployment scenarios.

---

## 📐 Latency Observations

- **Target Peak Latency:** < 10 ms (project requirement)  
- **Observed Peak Latency:**
  - **Apple Silicon (MPS Backend):**  
    Lower latency variance and more stable real-time behavior.
  - **NVIDIA RTX 3060 Ti (CUDA Backend):**  
    Higher raw throughput, but occasional peak latency spikes up to ~17 ms due to GPU scheduling and memory transfer overhead.

These results highlight a common real-world engineering trade-off:

> Higher computational power does not always guarantee lower or more stable latency.

---

## 🧠 Design Decisions on Latency Reporting

Instead of masking latency fluctuations, the system exposes them transparently through **live runtime metrics**.  
This design choice prioritizes **observability and system awareness** over idealized benchmark reporting.

Latency measurements directly informed:
- backend selection for real-time demonstrations,  
- hardware-specific optimization strategies,  
- and documentation of deployment constraints.

---

## ⚠️ Latency-Related Limitations

- Peak latency is influenced by GPU scheduling and driver-level behavior  
- Achieving consistent sub-10 ms latency is hardware- and platform-dependent  
- Real-world deployments may require additional platform-specific tuning  

These constraints are documented explicitly as part of a **transparent engineering approach**.

---


## 📄 CV-Ready Summary

> **Camera Occlusion Detection System**  
> Developed a real-time deep learning–based system to detect and classify camera occlusions by source (human, object, weather, sensor failure). Achieved over 90% classification accuracy with low-latency inference across heterogeneous hardware platforms (Apple MPS and NVIDIA GPUs). Focused on system robustness, real-time performance, and safety-critical deployment considerations.

---

## 🚀 Why This Project Matters

Most vision systems are optimized to answer a single question:

> *“What do I see?”*

This project addresses a more fundamental and often ignored problem:

> **“Should I trust what I see right now?”**

By continuously monitoring camera reliability and explicitly modeling different failure modes, the system adds a **critical layer of self-awareness** to vision-based systems.

In safety-critical applications, knowing *when not to trust perception* can be just as important as perception itself.

---
