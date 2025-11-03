# Change Detection of Scatterers Using Filtering-Based Temporal Noise Estimation in InSAR

This repository contains the source code for the Master's Thesis **“Change Detection in Multitemporal InSAR”**, submitted to **Gottfried Wilhelm Leibniz Universität Hannover**.

The project introduces and evaluates a **model-independent framework** for coherent change detection in InSAR time series, with a special focus on **Temporary Coherent Scatterers (TCS)**.

---

## 📜 Abstract

Reliable detection of structural change is crucial for infrastructure monitoring and hazard detection.  
This thesis investigates coherent change detection in multitemporal InSAR time series, focusing on **TCS**—scatterers that maintain coherence for only part of the observation period.

While traditional methods depend on displacement-based phase models (e.g., linear or polynomial fitting), they often fail in real-world scenarios with nonlinear deformation or abrupt events.

In response, this work introduces a **model-independent framework** that leverages **adaptive time-series filtering** to directly estimate phase noise from arc phase time series.  
By applying a **Likelihood Ratio Test (LRT)** to the filtered residuals, the method identifies change points without relying on deformation models.  
The thesis also provides a systematic comparison of this filtering-based approach against traditional model-based and spatial phase noise estimation methods.

---

## 🔬 Methodology

The core methodology of this thesis is a **model-independent pipeline** for TCS change detection:

1. **Point Selection**  
   Candidates are selected using the **Amplitude Dispersion Index (ADI)**.  
   - **First-order points (PS):** Highly stable points used for reference.  
   - **Second-order points (TCS):** Points with higher dispersion, which are the targets for change detection.

2. **Arc Phase Solution**  
   A **virtual reference** is created for each TCS candidate by averaging the complex values of its nearest first-order neighbors.  
   This provides a stable phase time series (the “arc phase”) that cancels out common atmospheric and orbital errors.

3. **Phase Noise Estimation**  
   The underlying deformation trend is separated from noise. This forms the core comparison of the thesis.  
   - **Proposed Method (Filtering):**  
     A model-free approach using adaptive temporal filters (e.g., **Butterworth** or **Savitzky–Golay**) to estimate the low-frequency deformation trend.  
     The residuals from this fit are treated as the **phase noise**.  
   - **Baseline Method (Linear Model):**  
     A traditional approach that fits a linear displacement model (velocity and DEM error) to the unwrapped phase and treats the residuals as noise.

4. **Change Detection**  
   A **Likelihood Ratio Test (LRT)** is applied to the estimated phase noise residuals.  
   It tests the hypothesis of a single change point by splitting the time series at every possible point and selecting the one that maximizes the likelihood of two distinct coherence states.

---

## 🧪 Experiments & Datasets

The framework was validated on two distinct datasets:

- **Spain Dataset:** A highway interchange in Valencia exhibiting significant **nonlinear deformation**, serving as a test case where traditional linear models are expected to fail.  
- **Miami Dataset:** The Surfside condominium collapse site in Miami, USA, representing an **abrupt structural failure**, providing a clear ground-truth event for validating detected change points.

---

## 📂 Project Structure

```
├── spatz_cli.py                      # Command-line interface for the SpaTZ framework
├── phase_noise_time.py               # Temporal phase noise estimation (filtering & model-based)
├── phase_noise_space.py              # Spatial phase noise estimation
├── arc_phase_plot.py                 # Interactive visualization of arc phases
├── arc_phase_temporal_noise_estimation.py # Evaluation of temporal filtering methods
├── compare_cd.py                     # Comparison between filtering and model-based CD
├── extract_scatterer_locations.py    # Exports detected scatterer locations with change dates
├── func.py                           # Common helper functions (SLC loading, ADI, virtual reference)
├── change_dates.py                   # Maps change indices to acquisition dates and exports CSV
└── data/                             # Input and output data folders
```

---

## 🧭 References

- **Hu, F. & Wu, J. (2021)** – Detecting spatio-temporal urban surface changes using identified temporary coherent scatterers. *JSEE, 32(6)*.  
- **Dörr, N., Schenk, A., & Hinz, S. (2022)** – Fully integrated temporary persistent scatterer interferometry. *IEEE TGRS, 60, 4412815*.  
- **Vavilla, S. V. (2025)** – Change Detection in Multitemporal InSAR, *Master’s Thesis*, Leibniz University Hannover.
