# Autonomous Runway Foreign Object Debris (FOD) Detection System

## Overview
Foreign Object Debris (FOD) on airport runways and taxiways, ranging from loose hardware and tools to tire fragments and wildlife, presents serious risks to aviation safety, including jet engine ingestion, tire blowouts, and airframe punctures.

This project develops an end-to-end computer vision solution engineered for autonomous runway inspection (deployable on unmanned aerial vehicles/drones and airfield patrol vehicles). The system prioritizes high debris recall to prevent life-threatening misses while controlling false positive rates on complex asphalt and concrete pavements.

---

## Quickstart: How to Run and Use the Model

### 1. Installation
Install the required packages:
```bash
pip install ultralytics torch
```

### 2. Model Location
The trained model weights are saved at:
`runs/detect/runs/detect/fod_yolo11n_runway/weights/best.pt`

### 3. How to Run Using Python
Use this simple script to run inference on an image, a video, or an entire directory:

```python
from ultralytics import YOLO

# 1. Load the trained model
model = YOLO("runs/detect/runs/detect/fod_yolo11n_runway/weights/best.pt")

# 2. Run inference (confidence tau = 0.25 optimized for runway safety)
results = model.predict(
    source="data/images/test/FOD-A_Dry_Bright_Hose_0005.jpg",  # can be image, video, or folder
    conf=0.25,      # Operating safety threshold (>=97% recall)
    imgsz=640,       # Input resolution
    save=True        # Save output image with bounding boxes drawn
)

# 3. Print detection details
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        print(f"Detected: {cls_name} (Confidence: {conf:.2f}) at BBox: {coords}")
```

### 4. How to Run from Terminal / Command Line (CLI)
You can run the model directly from PowerShell or Command Prompt:

- **Run on a single image:**
  ```bash
  yolo detect predict model="runs/detect/runs/detect/fod_yolo11n_runway/weights/best.pt" source="data/images/test/FOD-A_Dry_Bright_Hose_0005.jpg" conf=0.25 save=True
  ```

- **Run on a video:**
  ```bash
  yolo detect predict model="runs/detect/runs/detect/fod_yolo11n_runway/weights/best.pt" source="runway_fod_inspection_hd.mp4" conf=0.25 save=True
  ```

- **Run on a folder of images:**
  ```bash
  yolo detect predict model="runs/detect/runs/detect/fod_yolo11n_runway/weights/best.pt" source="data/images/test" conf=0.25 save=True
  ```

- **Run on a live webcam or drone video stream:**
  ```bash
  yolo detect predict model="runs/detect/runs/detect/fod_yolo11n_runway/weights/best.pt" source=0 conf=0.25 show=True
  ```

Annotated visual outputs are saved automatically in `runs/detect/predict/`.

### 5. Inputs and Outputs Summary
- **Input:** Images (`.jpg`, `.png`), video files (`.mp4`), or video streams. Images are resized to 640x640 pixels RGB.
- **Output:** Bounding box pixel coordinates `[x1, y1, x2, y2]`, classified into 6 hazard categories (`fastener`, `tool`, `tire_fragment`, `metal_scrap`, `plastic_debris`, `stone_rock`), with confidence probabilities from 0.0 to 1.0.

---

## Primary Engineering Challenges
1. **Extreme Scale Variance & Micro-Objects:** Objects range from tiny 10-30 px fasteners (screws, washers) to large tire treads across high-resolution runway frames.
2. **Safety-Critical Asymmetric Cost Function:**
   - **Miss (False Negative):** Catastrophic flight safety hazard.
   - **False Alarm (False Positive):** Disrupts flight schedules and causes operator alert fatigue.
   - Requires rigorous confidence threshold tuning favoring recall.
3. **Environmental Complexity:** Pavement joints, paint markings, tire skid rubber, oil stains, variable lighting (bright, dim, night), and surface reflections on wet tarmac.

---

## Target Object Classes & Taxonomy Reasoning
Mapped from 31 granular airfield debris categories into **6 FAA/ICAO-aligned operational hazard classes**:
- **`fastener`**: Bolts, nuts, screws, washers, rivets, nails (severe tire puncture / turbine blade hazard)
- **`tool`**: Wrenches, screwdrivers, pliers, cutters, hammers (structural impact / ingestion hazard)
- **`tire_fragment`**: Shredded tire rubber chunks, hose fragments
- **`metal_scrap`**: Safety wire, metal brackets, clamps, soda cans, batteries
- **`plastic_debris`**: Plastic bottles, luggage tags/wheels, packaging, pens
- **`stone_rock`**: Pavement chunks, gravel ballast, paint chips, wood

> **Why this grouping?**
> 1. **Data Sparsity:** In the raw 31-class dataset, classes like `FuelCap` or `ClampPart` had fewer than 15 samples, causing model divergence.
> 2. **Operational Reality:** Airport ground teams (FAA AC 150/5210-24) dispatch responses based on physical hazard mechanisms (puncture vs. structural impact vs. engine ingestion), not minor hardware sub-specifications.

---

## Dataset & Exploratory Data Analysis (EDA)
- **Benchmark Source:** FOD-A (Foreign Object Debris in Airports) benchmark.
- **Why FOD-A:** Real airfield asphalt/concrete pavements under 6 authentic conditions (dry/wet tarmac, bright/dim/dusk lighting) captured from runway camera rigs and UAS drone perspectives.
- **Data Stratification & Splits:** 4,000 curated, non-redundant images partitioned into:
  - **Train:** 2,800 images (70%)
  - **Validation:** 600 images (15%)
  - **Held-out Test:** 600 images (15%)
- **Multi-Condition Coverage:** Spans dry and wet asphalt, with bright, dim, and dark lighting conditions.
- **Pipeline Implementation:** Fully reproducible Jupyter notebook in [`01_EDA_and_Data_Preparation.ipynb`](./01_EDA_and_Data_Preparation.ipynb) with YOLO format labels and [`dataset.yaml`](./data/dataset.yaml).

---

## Model Selection & Training Strategy
- **Detector Architecture:** **YOLO11 Nano (`yolo11n`)** with C2PSA (Cross-Stage Partial with Spatial Attention).
- **Why YOLO11 Nano:**
  - **C2PSA Spatial Attention:** Acts as a visual spotlight, picking out metallic glints and tool edges amidst 95%+ monotonous tarmac texture.
  - **Ultra-Lightweight & Fast:** 2.59M parameters (22% smaller than YOLOv8n) running at **4.3 ms (~150+ FPS)**, ideal for sub-15W drone onboard companion computers (Jetson Orin Nano).
  - **Outperforms Alternatives:** Avoids the heavy compute overhead of Faster R-CNN (>85 ms) and RT-DETR (>45 ms).
- **Input Resolution:** $640 \times 640$ (preserves feature representations for micro-fasteners $< 32\text{ px}$).
- **Batch Size:** 16 (optimized for local 4GB RTX 2050 GPU with automatic mixed precision `fp16`).
- **Optimization:** AdamW optimizer with Cosine Annealing learning rate schedule.
- **Airport-Specific Augmentations:** Mosaic (simulates altitude/zoom changes), HSV color jitter (sunlight, shadows, dusk), and rotational invariance.

---

## Model Training Results (Task 2)
- **Trained Model Checkpoint:** `runs/detect/runs/detect/fod_yolo11n_runway/weights/best.pt` (5.5 MB)
- **Training Time:** 35 epochs completed in ~27 minutes on NVIDIA GeForce RTX 2050 GPU.
- **Inference Speed:** **4.3 ms per image (~150+ FPS)** on GPU, fully satisfying real-time UAV flight requirements.
- **Validation Metrics on `best.pt`:**
  - **Overall Precision ($P$):** **94.8%**
  - **Overall Recall ($R$):** **95.7%** (Critical safety benchmark: $<4.3\%$ debris miss rate)
  - **$mAP@50$:** **97.8%**
  - **$mAP@50\text{-}95$:** **74.8%**
- **Pipeline Implementation:** Complete training setup documented in [`02_Model_Training_Pipeline.ipynb`](./02_Model_Training_Pipeline.ipynb).

---

## Held-Out Test Evaluation & Metrics (Tasks 3 & 5)
Evaluated strictly on **600 quarantined test images**:
- **Test Precision ($P$):** **95.34%**
- **Test Recall ($R$):** **95.77%**
- **Test $mAP@50$:** **96.81%**
- **Test $mAP@50\text{-}95$:** **71.53%**
- **Detection Breakdown:** 595 True Positives, 75 False Positives (False Alarms), 18 False Negatives (Missed Objects).
- **Runway Safety Threshold Tuning:** Evaluated precision-recall trade-off across confidence levels; identified **$\tau = 0.25$ as optimal operating point**, yielding **$97.06\%$ hazard recall** while preventing operator alert fatigue ($88.81\%$ precision).

---

## Submission Deliverables & Artifacts
1. **Full Technical Report:** [`FOD_Detection_Technical_Report.md`](./FOD_Detection_Technical_Report.md) (and compiled PDF in `submission/FOD_Detection_Technical_Report.pdf`).
2. **Result Inspection Video:** [`runway_fod_inspection_hd.mp4`](./runway_fod_inspection_hd.mp4) (24.0s 720p HD video @ 30 FPS, showcasing real-time detections with corner reticles, confidence scores, and dual-panel UAV telemetry HUD).
3. **6 Diagnostic Images:** Located in [`diagnostic_results/`](./diagnostic_results/):
   - 2 True Positive cases (accurate debris localization).
   - 2 False Positive cases (false alarms on asphalt cracks/textures).
   - 2 False Negative cases (misses on micro-scale / camouflaged hazards).
   - Consolidated 2x3 overview in [`diagnostic_summary_grid.png`](./diagnostic_results/diagnostic_summary_grid.png).
4. **Reproducible Jupyter Notebooks:**
   - [`01_EDA_and_Data_Preparation.ipynb`](./01_EDA_and_Data_Preparation.ipynb)
   - [`02_Model_Training_Pipeline.ipynb`](./02_Model_Training_Pipeline.ipynb)
   - [`03_Evaluation_and_Error_Analysis.ipynb`](./03_Evaluation_and_Error_Analysis.ipynb)
