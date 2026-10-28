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


