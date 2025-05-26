---
layout: single
title: "FLORO: Foundation Learning Of Remote Sensing Observations for Ecological Research"
permalink: /floro/
author_profile: true
---
<p style="text-align: center;">
Jorge L. Rodríguez, Kasper Johansen, Areej Alwahas, Victor Angulo-Morales, Mariana Elías-Lara, Matthew F. McCabe
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

![FLORO pretraining recosntruction](assets/gifs/floro_pretraining.gif)

---

## **Overview**

**FLORO (Foundation Learning Of Remote Sensing Observations for Ecological Research)** is a multimodal, multitask Vision Transformer-based foundation model designed for scalable ecological monitoring using both multispectral satellite data and elevation sources.

Built with a masked autoencoder backbone and fine-tuned on diverse ecological tasks, FLORO generalizes well across modalities, sensor types, and ecosystems.

---

## 🔍 **Highlights**

- **Multimodal Inputs**: Uses multispectral bands + digital surface models (DSM) from SRTM, UAV, or photogrammetric data.
- **Self-Supervised Pretraining**: Trained with adaptive masked autoencoding strategies across 400K+ image patches.
- **Flexible Decoders**: Task-specific decoders output vegetation structure (CHM), forage biomass, and nutrient content.

---

## 📄 **Abstract**

While biodiversity modeling has benefited greatly from the availability of diverse remote sensing observations, these models are often limited by low accuracies and computational complexity. Recent developments in machine and deep learning techniques provide a potential pathway to overcome such constraints. Here we present a foundation model with an architecture that integrates a multimodal masked auto-encoder, a Vision Transformer, and multiple decoders, to process high-resolution satellite imagery, with an initial goal to enhance vegetation description and identification in dryland ecosystems. The Foundation Learning for Optical Remote Sensing Observations (FLORO) approach was developed in part to provide a cost-effective solution for large-scale ecological studies. The model architecture facilitates simultaneous learning from multimodal remote sensing data, leading to more accurate and computationally efficient monitoring of biodiversity. Here we detail the development and validation of the foundation model, its potential applications in ecological monitoring, and explore how it addresses current barriers to ecological monitoring in drylands. We validated FLORO on the ISPRS Potsdam dataset, achieving an overall accuracy of 95.4%, outperforming the previous best model by 3.5%. Class-wise, FLORO attained accuracies of 92.43% for Trees, 90.11% for Low Vegetation, and 99.22% for Cars, demonstrating its capability to accurately identify both complex vegetative structures and fine-scale objects. In UAV-based ecological monitoring tasks, fine-tuning both encoder and decoder yielded 97.49\% overall accuracy, a 70.56% mean F1 score, and a 67.64% mean IoU, with minimal performance loss even when trained with noisy labels. Overall, these results demonstrate FLORO’s potential as a foundation model for large-scale, multimodal ecological applications, particularly in data-scarce and environmentally challenging landscapes such as drylands.

---

## 🧠 **Architecture**

> A full Transformer-based encoder is pretrained via masked image autoencoding. Downstream decoders are optimized using supervised objectives for both segmentation and regression tasks.

![FLORO architecture](assets/images/floro_architecture.png)
