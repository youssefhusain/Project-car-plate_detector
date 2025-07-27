

###  **README (Professional Version)**

**Project Title**: **Smart Traffic Scene Analysis and License Plate Recognition System**

**Overview**:
This system leverages deep learning and computer vision techniques to detect and track vehicles, recognize license plates, and describe traffic scenes using natural language. It integrates multiple AI models, including YOLO for object detection, EasyOCR for license plate reading, and Qwen-VL for visual-language scene interpretation.

---

###  Features

* **Real-time Vehicle Detection & Tracking**
  Uses YOLOv8 and a SORT-based Kalman filter tracker to detect and track vehicles across frames.

* **License Plate Recognition**
  Applies EasyOCR with character correction mapping to extract and normalize license plate text.

* **Scene Understanding**
  Utilizes Qwen2.5-VL to generate descriptive captions of traffic scenes from video frames.

* **Structured Output**
  Extracted data is saved in a clean JSON format including vehicle bounding boxes, license plate text, scores, and scene descriptions.

---

###  Models Used

* `YOLOv8` – for vehicle and license plate detection.
* `EasyOCR` – for text extraction from plates.
* `Qwen2.5-VL` – for image-to-text traffic scene description.
* `SORT` – for tracking vehicles across frames.

---

### Use Cases

* Smart city surveillance
* Automated traffic law enforcement
* Data collection for urban planning
* Vehicle tracking in security systems

---

###  Description (Mini Version for Showcase)

> A multi-model AI pipeline that detects and tracks vehicles, recognizes license plates using OCR with post-processing, and describes the traffic scene in natural language. Integrates YOLO, EasyOCR, and Qwen-VL with real-time video input and outputs structured JSON reports for intelligent traffic monitoring.


