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
