---
layout: single
title: "FLORO: A Multimodal Geospatial Foundation Model for Ecological Remote Sensing Across Sensors and Scales"
permalink: /floro/
author_profile: true
---

<p style="text-align: center;">
Jorge L. Rodríguez, Victor Angulo-Morales, Areej Alwahas, Mariana Elías-Lara, Kasper Johansen, Fernando T. Maestre and Matthew F. McCabe
</p>

<p style="text-align: center;"><i>
Climate and Livability Initiative, Biological and Environmental Science and Engineering Division, King Abdullah University of Science and Technology, Thuwal, Saudi Arabia</i>
</p>

<!-- Insert icons here -->
<div style="text-align: center; margin: 1.5rem 0;">
  <a href="#" title="Paper" class="external-link button is-normal is-rounded is-black" style="margin: 0 10px;">
  <span class="icon">
    <img src="/assets/icons/arxiv-logomark-small-white.svg" alt="Paper" width="24" height="24">
  </span>
  <span> Paper </span>
  </a>
  <a href="#" title="Code" class="external-link button is-normal is-rounded is-black" style="margin: 0 10px;">
  <span class="icon">
    <img src="/assets/icons/github-mark-white.svg" alt="GitHub" width="24" height="24">
  </span>
  <span> Code </span>
  </a>
</div>

---

## **Overview**

**FLORO:**, is a multimodal geospatial foundation model designed to learn transferable representations from a compact but highly diverse remote sensing corpus.

FLORO is pretrained using masked autoencoding on heterogeneous Earth observation data, including multispectral satellite imagery, synthetic-aperture radar, high-resolution commercial imagery, UAV products, and terrain information. Rather than relying only on very large pretraining corpora, FLORO explores whether diversity across sensors, spatial resolutions, spectral configurations, and ecological settings can produce representations that transfer effectively across remote sensing tasks.

Under the frozen-encoder evaluation protocol of the PANGAEA benchmark, FLORO demonstrates strong transfer performance across semantic segmentation, scene classification, and regression tasks. Its results suggest that competitive geospatial representations can emerge from carefully curated multimodal pretraining data, even when the number of pretraining samples is much smaller than in several recent large-scale remote sensing foundation models.

---

## 🔍 **Highlights**

- **Compact but diverse pretraining corpus**: FLORO is pretrained on approximately 80K remote sensing image samples spanning satellite, airborne, UAV, optical, SAR, and terrain-derived data sources.

- **Multimodal Earth observation inputs**: The model supports multispectral optical data, Sentinel-1 SAR backscatter, elevation products, UAV-derived structural information, and high-resolution imagery.

- **Masked autoencoding for heterogeneous data**: FLORO learns by reconstructing masked content from partially observed multimodal inputs, encouraging the encoder to capture both spectral and spatial structure.

- **Flexible handling of missing modalities**: Availability and validity channels allow the model to distinguish between observed, missing, and invalid data, supporting transfer across datasets with different sensor configurations.

- **Strong frozen-encoder transfer**: In standardized PANGAEA evaluations, FLORO remains competitive across segmentation, scene classification, and regression tasks using benchmark-defined downstream decoders.

- **Ecological monitoring focus**: FLORO is designed for applications where structural, spectral, and environmental information are jointly important, including vegetation mapping, biomass estimation, canopy-height reconstruction, and ecosystem monitoring.

---

## 🧠 **Architecture**

FLORO uses a Vision Transformer encoder pretrained through masked autoencoding. During pretraining, heterogeneous remote sensing observations are tokenized and partially masked. The encoder learns latent representations from the visible tokens, while lightweight reconstruction decoders predict the masked content for each modality.

After pretraining, the shallow reconstruction decoders are discarded. The pretrained encoder is then evaluated under frozen-encoder transfer, where task-specific benchmark decoders are trained for downstream tasks such as semantic segmentation, scene classification, and regression.

![FLORO architecture](assets/images/floro_architecture.png)

---

## 🌍 **Pretraining Data**

FLORO is pretrained on a heterogeneous collection of remote sensing observations designed to expose the model to variation in spatial resolution, spectral coverage, sensing geometry, and ecological context. The pretraining corpus includes:

- **Sentinel-2 multispectral imagery**, providing medium-resolution optical observations across visible, red-edge, near-infrared, and shortwave-infrared bands.
- **Sentinel-1 SAR imagery**, providing radar backscatter information that complements optical observations, particularly under cloud cover or degraded optical conditions.
- **SkySat high-resolution imagery**, introducing fine spatial detail and commercial satellite observation characteristics.
- **Terrain and elevation products**, including global and local digital elevation or terrain-derived data.
- **UAV RGB and multispectral products**, including very high-resolution orthomosaics, digital surface models, and vegetation-structure information.

The objective is not only to increase the number of pretraining samples, but to increase the diversity of sensing conditions encountered during representation learning.

---

## 🧪 **Evaluation**

FLORO is evaluated using the **PANGAEA** benchmark, a standardized framework for assessing the transferability of geospatial foundation models across diverse Earth observation tasks.

The evaluation follows a frozen-encoder protocol: the pretrained FLORO encoder remains fixed, while benchmark-defined downstream decoders are trained for each task. This setting isolates the quality and transferability of the learned representation, rather than measuring the performance ceiling obtainable through full end-to-end fine-tuning.

FLORO is evaluated across three broad task families:

- **Semantic segmentation**, including land-cover, flood, burn-scar, agricultural, and urban-mapping benchmarks.
- **Scene classification**, including multispectral land-use and land-cover classification.
- **Regression**, including ecological and vegetation-structure tasks such as biomass estimation and canopy-height reconstruction.

Across these evaluations, FLORO achieves strong transfer performance despite being pretrained on substantially fewer image samples than several recent large-scale geospatial foundation models.

---

## 📊 **Why FLORO?**

Many geospatial foundation models scale primarily by increasing the size of the pretraining corpus. FLORO explores a complementary direction: learning from a smaller but highly heterogeneous collection of remote sensing observations.

This design is motivated by ecological remote sensing, where downstream tasks often require generalization across sensors, landscapes, resolutions, and data availability conditions. In these settings, robustness to heterogeneous inputs can be as important as raw pretraining scale.

FLORO therefore emphasizes:

- sensor diversity,
- modality flexibility,
- ecological relevance,
- transferability under limited downstream adaptation,
- and compatibility with standardized benchmark evaluation.

---

## 🌱 **Applications**

FLORO is being developed for remote sensing applications where ecological structure and function need to be inferred across space and scale. Potential applications include:

- vegetation and land-cover mapping,
- canopy-height estimation,
- aboveground biomass estimation,
- ecological condition assessment,
- dryland and rangeland monitoring,
- multimodal data fusion,
- and transfer learning for remote sensing datasets with limited labels.

---

## 📌 **Status**

The current version of FLORO has been evaluated under the PANGAEA benchmark using a frozen-encoder transfer protocol. The model has been tested across semantic segmentation, scene classification, and regression tasks, showing strong generalization across multiple datasets and sensing configurations.

Code, pretrained weights, and manuscript links will be added once publicly available.