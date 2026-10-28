# Autonomous Runway Foreign Object Debris (FOD) Detection System

## Overview
Foreign Object Debris (FOD) on airport runways and taxiways—ranging from loose hardware and tools to tire fragments and wildlife—presents catastrophic risks to aviation safety, including jet engine ingestion, tire blowouts, and airframe punctures.

This project develops an end-to-end computer vision solution engineered for autonomous runway inspection (deployable on unmanned aerial vehicles/drones and airfield patrol vehicles). The system prioritizes high debris recall to prevent life-threatening misses while controlling false positive rates on complex asphalt and concrete pavements.

---

## Primary Engineering Challenges
1. **Extreme Scale Variance & Micro-Objects:** Objects range from tiny 10–30 px fasteners (screws, washers) to large tire treads across high-resolution runway frames.
2. **Safety-Critical Asymmetric Cost Function:**
   - **Miss (False Negative):** Catastrophic flight safety hazard.
   - **False Alarm (False Positive):** Disrupts flight schedules and causes operator alert fatigue.
   - Requires rigorous confidence threshold tuning favoring recall.
3. **Environmental Complexity:** Pavement joints, paint markings, tire skid rubber, oil stains, variable lighting (bright, dim, night), and surface reflections on wet tarmac.

---

## Target Object Classes
Mapped from 31 granular airfield debris categories into **6 FAA/ICAO-aligned operational hazard classes**:
- **`fastener`**: Bolts, nuts, screws, washers, rivets, nails (severe tire puncture / turbine blade hazard)
- **`tool`**: Wrenches, screwdrivers, pliers, cutters, hammers (structural impact / ingestion hazard)
- **`tire_fragment`**: Shredded tire rubber chunks, hose fragments
- **`metal_scrap`**: Safety wire, metal brackets, clamps, soda cans, batteries
- **`plastic_debris`**: Plastic bottles, luggage tags/wheels, packaging, pens
- **`stone_rock`**: Pavement chunks, gravel ballast, paint chips, wood

---

## Dataset & Exploratory Data Analysis (EDA)
- **Benchmark Source:** FOD-A (Foreign Object Debris in Airports) airfield dataset.
- **Taxonomy:** 31 raw categories mapped into 6 operational runway hazard classes (`fastener`, `tool`, `tire_fragment`, `metal_scrap`, `plastic_debris`, `stone_rock`).
- **Data Stratification & Splits:** 4,000 curated, non-redundant images partitioned into:
  - **Train:** 2,800 images (70%)
  - **Validation:** 600 images (15%)
  - **Held-out Test:** 600 images (15%)
- **Multi-Condition Coverage:** Spans dry and wet asphalt, with bright, dim, and dark lighting conditions.
- **Pipeline Implementation:** Fully reproducible Jupyter notebook in [`01_EDA_and_Data_Preparation.ipynb`](./01_EDA_and_Data_Preparation.ipynb) with YOLO format labels and [`dataset.yaml`](./data/dataset.yaml).

---

## Model Selection & Training Strategy
- **Detector Architecture:** **YOLO11 Nano (`yolo11n`)** with C2PSA (Cross-Stage Partial with Spatial Attention), engineered to focus on small anomalous debris amidst vast tarmac background.
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
1. **Full Technical Report:** [`FOD_Detection_Technical_Report.md`](./FOD_Detection_Technical_Report.md) (covers dataset, training justification, metrics, error analysis, threshold reasoning, and UAV edge architecture).
2. **Result Inspection Video:** [`runway_fod_inspection_hd.mp4`](./runway_fod_inspection_hd.mp4) (24.0s 720p HD cinematic video @ 30 FPS, showcasing real-time detections with corner reticles, confidence scores, and dual-panel UAV telemetry HUD).
3. **6 Diagnostic Images:** Located in [`diagnostic_results/`](./diagnostic_results/):
   - 2 True Positive cases (accurate debris localization).
   - 2 False Positive cases (false alarms on asphalt cracks/textures).
   - 2 False Negative cases (misses on micro-scale / camouflaged hazards).
   - Consolidated 2x3 overview in [`diagnostic_summary_grid.png`](./diagnostic_results/diagnostic_summary_grid.png).
4. **Reproducible Jupyter Notebooks:**
   - [`01_EDA_and_Data_Preparation.ipynb`](./01_EDA_and_Data_Preparation.ipynb)
   - [`02_Model_Training_Pipeline.ipynb`](./02_Model_Training_Pipeline.ipynb)
   - [`03_Evaluation_and_Error_Analysis.ipynb`](./03_Evaluation_and_Error_Analysis.ipynb)
