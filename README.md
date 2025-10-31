# Change-detection-of-scatterers-using-filtering-based-temporal-noise-estimation-in-InSAR

This repository contains the source code for the Master's Thesis, "Change Detection in Multitemporal InSAR", submitted to the Gottfried Wilhelm Leibniz Universität Hannover.

The project introduces and evaluates a model-independent framework for coherent change detection in InSAR time series, with a special focus on **Temporary Coherent Scatterers (TCS)

## 📜 Abstract

Reliable detection of structural change is crucial for infrastructure monitoring and hazard detection. This thesis investigates coherent change detection in multitemporal InSAR time series, focusing on TCS—scatterers that maintain coherence for only part of the observation period[span_12](end_span). [span_13](start_span)While traditional methods depend on displacement-based phase models (e.g., linear or polynomial fitting), they often fail in real-world scenarios with nonlinear deformation or abrupt events[span_13](end_span).

[span_14](start_span)In response, this work introduces a **model-independent framework** that leverages adaptive time-series filtering to directly estimate phase noise from arc phase time series[span_14](end_span). [span_15](start_span)By applying a **Likelihood Ratio Test (LRT)** to the filtered residuals, the method identifies change points without relying on deformation models[span_15](end_span). [span_16](start_span)The thesis provides a systematic comparison of this filtering-based approach against traditional model-based and spatial phase noise estimation methods[span_16](end_span).

## 🔬 Methodology

The core methodology of this thesis is a model-independent pipeline for TCS change detection:

1.  **[span_17](start_span)Point Selection:** Candidates are selected using the **Amplitude Dispersion Index (ADI)**[span_17](end_span).
    * **[span_18](start_span)First-order points (PS):** Highly stable points used for reference[span_18](end_span).
    * **[span_19](start_span)Second-order points (TCS):** Points with higher dispersion, which are the targets for change detection[span_19](end_span).
2.  **[span_20](start_span)Arc Phase Solution:** A **virtual reference** is created for each TCS candidate by averaging the complex values of its nearest first-order neighbors[span_20](end_span). [span_21](start_span)[span_22](start_span)This provides a stable phase time series (the "arc phase") that cancels out common atmospheric and orbital errors[span_21](end_span)[span_22](end_span).
3.  **Phase Noise Estimation:** The underlying deformation trend is separated from noise. This is the core comparison of the thesis.
    * **[span_23](start_span)[span_24](start_span)Proposed Method (Filtering):** A model-free approach using adaptive temporal filters (like **Butterworth** or **Savitzky-Golay**) to model the low-frequency deformation trend[span_23](end_span)[span_24](end_span). [span_25](start_span)The residuals from this fit are treated as the phase noise[span_25](end_span).
    * **[span_26](start_span)[span_27](start_span)Baseline Method (Linear Model):** A traditional approach that fits a linear displacement model (velocity and DEM error) to the unwrapped phase and treats the residuals as noise[span_26](end_span)[span_27](end_span).
4.  **[span_28](start_span)Change Detection:** A **Likelihood Ratio Test (LRT)** is applied to the estimated phase noise residuals[span_28](end_span). [span_29](start_span)[span_30](start_span)It tests the hypothesis of a single change point by splitting the time series at every possible point and finding the split that maximizes the statistical likelihood of two distinct coherence states[span_29](end_span)[span_30](end_span).

## 🧪 Experiments & Datasets

The framework was validated on two distinct datasets:

* **[span_31](start_span)Spain Dataset:** A highway interchange in Valencia[span_31](end_span). [span_32](start_span)[span_33](start_span)This area exhibits significant **nonlinear deformation**, serving as a test case where traditional linear models are expected to fail[span_32](end_span)[span_33](end_span).
* **[span_34](start_span)Miami Dataset:** The Surfside condominium in Miami, USA[span_34](end_span). [span_35](start_span)[span_36](start_span)This site experienced a catastrophic **abrupt collapse** and provides a clear ground-truth event for validating the detected change point[span_35](end_span)[span_36](end_span).

[span_37](start_span)
[span_37](end_span)

## 📂 Project Structure

