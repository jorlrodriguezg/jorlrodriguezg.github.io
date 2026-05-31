---
layout: single
title: "FLORO: A Multimodal Geospatial Foundation Model for Ecological Remote Sensing Across Sensors and Scales"
permalink: /floro/
author_profile: true
---

<style>
/* ==========================================================
   FLORO project page styling
   Scoped to avoid affecting the rest of the website
   ========================================================== */

.floro-page {
  --floro-dark: #17202a;
  --floro-text: #2f3a45;
  --floro-muted: #64748b;
  --floro-green: #2f855a;
  --floro-blue: #2563eb;
  --floro-gold: #b7791f;
  --floro-bg: #f8fafc;
  --floro-card: #ffffff;
  --floro-border: #e2e8f0;
  --floro-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
  --floro-radius: 18px;

  color: var(--floro-text);
  line-height: 1.72;
}

.floro-page p {
  font-size: 1.02rem;
}

.floro-page .justify {
  text-align: justify;
  text-justify: inter-word;
}

.floro-hero {
  background: linear-gradient(135deg, #eef7f1 0%, #f8fafc 45%, #eef4ff 100%);
  border: 1px solid var(--floro-border);
  border-radius: 24px;
  padding: 2.2rem 2rem;
  margin: 1.5rem 0 2rem 0;
  box-shadow: var(--floro-shadow);
}

.floro-hero h1 {
  margin-top: 0;
  margin-bottom: 0.75rem;
  font-size: clamp(1.8rem, 3vw, 2.7rem);
  line-height: 1.15;
  color: var(--floro-dark);
}

.floro-hero .subtitle {
  font-size: 1.15rem;
  color: var(--floro-muted);
  max-width: 920px;
  margin-bottom: 1.3rem;
}

.floro-authors,
.floro-affiliations {
  text-align: center;
  color: var(--floro-muted);
}

.floro-authors {
  font-size: 0.98rem;
  margin-top: 1rem;
  margin-bottom: 0.6rem;
}

.floro-affiliations {
  font-size: 0.9rem;
  line-height: 1.5;
}

.floro-buttons {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.8rem;
  margin: 1.6rem 0 0.4rem 0;
}

.floro-button {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.65rem 1rem;
  border-radius: 999px;
  background: #111827;
  color: #ffffff !important;
  text-decoration: none !important;
  font-weight: 600;
  font-size: 0.92rem;
  transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}

.floro-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 20px rgba(17, 24, 39, 0.18);
  background: #000000;
}

.floro-button img {
  width: 21px;
  height: 21px;
}

.floro-stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(130px, 1fr));
  gap: 1rem;
  margin: 1.8rem 0 2.2rem 0;
}

.floro-stat {
  background: var(--floro-card);
  border: 1px solid var(--floro-border);
  border-radius: var(--floro-radius);
  padding: 1rem;
  text-align: center;
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
}

.floro-stat .number {
  display: block;
  font-size: 1.45rem;
  font-weight: 800;
  color: var(--floro-dark);
}

.floro-stat .label {
  display: block;
  margin-top: 0.25rem;
  font-size: 0.86rem;
  color: var(--floro-muted);
}

.floro-section {
  margin: 2.3rem 0;
}

.floro-section h2 {
  color: var(--floro-dark);
  font-size: 1.55rem;
  margin-bottom: 0.9rem;
  padding-bottom: 0.45rem;
  border-bottom: 2px solid #e5e7eb;
}

.floro-panel {
  background: var(--floro-bg);
  border: 1px solid var(--floro-border);
  border-radius: var(--floro-radius);
  padding: 1.4rem 1.5rem;
  margin: 1.1rem 0;
}

.floro-highlight-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(240px, 1fr));
  gap: 1rem;
  margin-top: 1.2rem;
}

.floro-card {
  background: var(--floro-card);
  border: 1px solid var(--floro-border);
  border-radius: var(--floro-radius);
  padding: 1.15rem 1.2rem;
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
}

.floro-card h3 {
  margin-top: 0;
  margin-bottom: 0.35rem;
  font-size: 1.02rem;
  color: var(--floro-dark);
}

.floro-card p {
  margin-bottom: 0;
  color: var(--floro-muted);
  font-size: 0.95rem;
  line-height: 1.62;
}

.floro-list {
  list-style: none;
  padding-left: 0;
  margin-left: 0;
}

.floro-list li {
  position: relative;
  padding-left: 1.8rem;
  margin-bottom: 0.8rem;
}

.floro-list li::before {
  content: "✦";
  position: absolute;
  left: 0;
  top: 0;
  color: var(--floro-green);
  font-weight: 700;
}

.floro-figure {
  margin: 1.6rem 0 2rem 0;
  background: #ffffff;
  border: 1px solid var(--floro-border);
  border-radius: 20px;
  padding: 0.9rem;
  box-shadow: var(--floro-shadow);
}

.floro-figure img {
  width: 100%;
  border-radius: 14px;
  display: block;
}

.floro-caption {
  color: var(--floro-muted);
  font-size: 0.9rem;
  margin: 0.75rem 0 0 0;
  text-align: center;
}

.floro-quote {
  border-left: 4px solid var(--floro-green);
  background: #f0fdf4;
  padding: 1rem 1.2rem;
  border-radius: 0 14px 14px 0;
  color: #2f3a45;
  margin: 1.2rem 0;
}

.floro-two-column {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.2rem;
  align-items: start;
}

.floro-status {
  background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
  border: 1px solid var(--floro-border);
  border-radius: var(--floro-radius);
  padding: 1.3rem 1.5rem;
}

@media (max-width: 800px) {
  .floro-stats,
  .floro-highlight-grid,
  .floro-two-column {
    grid-template-columns: 1fr;
  }

  .floro-hero {
    padding: 1.6rem 1.2rem;
  }

  .floro-page .justify {
    text-align: left;
  }
}
</style>

<div class="floro-page">

<section class="floro-hero">

<h1>FLORO: A Multimodal Geospatial Foundation Model for Ecological Remote Sensing Across Sensors and Scales</h1>

<p class="subtitle">
A compact but diverse foundation model for learning transferable representations from heterogeneous Earth observation data, with a focus on ecological and environmental remote sensing.
</p>

<div class="floro-authors">
Jorge L. Rodríguez, Victor Angulo-Morales, Areej Alwahas, Mariana Elías-Lara, Fida Thoker, Kasper Johansen, Bernard Ghanem, Fernando T. Maestre, and Matthew F. McCabe
</div>

<div class="floro-affiliations">
<em>Biological and Environmental Science and Engineering Division, King Abdullah University of Science and Technology, Thuwal, Saudi Arabia</em><br>
<em>Computer, Electrical and Mathematical Science and Engineering Division, King Abdullah University of Science and Technology, Thuwal, Saudi Arabia</em>
</div>

<div class="floro-buttons">
  <a href="https://arxiv.org/abs/2605.28174" title="Paper" class="floro-button">
    <img src="/assets/icons/arxiv-logomark-small-white.svg" alt="arXiv icon">
    <span>Paper</span>
  </a>

  <a href="#" title="Code" class="floro-button">
    <img src="/assets/icons/github-mark-white.svg" alt="GitHub icon">
    <span>Code</span>
  </a>
</div>

</section>

<div class="floro-stats">
  <div class="floro-stat">
    <span class="number">~80K</span>
    <span class="label">pretraining samples</span>
  </div>

  <div class="floro-stat">
    <span class="number">Multi-sensor</span>
    <span class="label">satellite, UAV, SAR, terrain</span>
  </div>

  <div class="floro-stat">
    <span class="number">PANGAEA</span>
    <span class="label">standardized evaluation</span>
  </div>

  <div class="floro-stat">
    <span class="number">Frozen encoder</span>
    <span class="label">transfer protocol</span>
  </div>
</div>

<section class="floro-section">

<h2>Overview</h2>

<div class="floro-panel">

<p class="justify">
<strong>FLORO</strong> is a multimodal geospatial foundation model designed to learn transferable representations from a compact but highly diverse remote sensing corpus.
</p>

<p class="justify">
FLORO is pretrained using masked autoencoding on heterogeneous Earth observation data, including multispectral satellite imagery, synthetic-aperture radar, high-resolution commercial imagery, UAV products, and terrain information. Rather than relying only on very large pretraining corpora, FLORO explores whether diversity across sensors, spatial resolutions, spectral configurations, and ecological settings can produce representations that transfer effectively across remote sensing tasks.
</p>

<p class="justify">
Under the frozen-encoder evaluation protocol of the PANGAEA benchmark, FLORO demonstrates strong transfer performance across semantic segmentation, scene classification, and regression tasks. Its results suggest that competitive geospatial representations can emerge from carefully curated multimodal pretraining data, even when the number of pretraining samples is much smaller than in several recent large-scale remote sensing foundation models.
</p>

</div>

</section>

<section class="floro-section">

<h2>🔍 Highlights</h2>

<div class="floro-highlight-grid">

<div class="floro-card">
<h3>Compact but diverse pretraining</h3>
<p>Pretrained on approximately 80K remote sensing image samples spanning satellite, airborne, UAV, optical, SAR, and terrain-derived data sources.</p>
</div>

<div class="floro-card">
<h3>Multimodal Earth observation inputs</h3>
<p>Supports multispectral optical data, Sentinel-1 SAR backscatter, elevation products, UAV-derived structural information, and high-resolution imagery.</p>
</div>

<div class="floro-card">
<h3>Masked autoencoding</h3>
<p>Learns by reconstructing masked content from partially observed multimodal inputs, encouraging the encoder to capture spectral and spatial structure.</p>
</div>

<div class="floro-card">
<h3>Missing-modality awareness</h3>
<p>Availability and validity channels allow the model to distinguish between observed, missing, and invalid data across different sensor configurations.</p>
</div>

<div class="floro-card">
<h3>Strong frozen-encoder transfer</h3>
<p>In standardized PANGAEA evaluations, FLORO remains competitive across segmentation, scene classification, and regression tasks.</p>
</div>

<div class="floro-card">
<h3>Ecological monitoring focus</h3>
<p>Designed for applications where structural, spectral, and environmental information are jointly important, including vegetation mapping and ecosystem monitoring.</p>
</div>

</div>

</section>

<section class="floro-section">

<h2>🧠 Architecture</h2>

<p class="justify">
FLORO uses a Vision Transformer encoder pretrained through masked autoencoding. During pretraining, heterogeneous remote sensing observations are tokenized and partially masked. The encoder learns latent representations from the visible tokens, while lightweight reconstruction decoders predict the masked content for each modality.
</p>

<p class="justify">
After pretraining, the shallow reconstruction decoders are discarded. The pretrained encoder is then evaluated under frozen-encoder transfer, where task-specific benchmark decoders are trained for downstream tasks such as semantic segmentation, scene classification, and regression.
</p>

<figure class="floro-figure">
  <img src="assets/images/floro_architecture.png" alt="FLORO architecture">
  <figcaption class="floro-caption">
    FLORO architecture. Heterogeneous multimodal observations are tokenized, masked, encoded, and reconstructed during pretraining.
  </figcaption>
</figure>

</section>

<section class="floro-section">

<h2>🌍 Pretraining Data</h2>

<p class="justify">
FLORO is pretrained on a heterogeneous collection of remote sensing observations designed to expose the model to variation in spatial resolution, spectral coverage, sensing geometry, and ecological context.
</p>

<div class="floro-two-column">

<div class="floro-card">
<h3>Data sources</h3>
<ul class="floro-list">
  <li><strong>Sentinel-2 multispectral imagery</strong>, providing medium-resolution optical observations across visible, red-edge, near-infrared, and shortwave-infrared bands.</li>
  <li><strong>Sentinel-1 SAR imagery</strong>, providing radar backscatter information that complements optical observations.</li>
  <li><strong>SkySat high-resolution imagery</strong>, introducing fine spatial detail and commercial satellite observation characteristics.</li>
  <li><strong>Terrain and elevation products</strong>, including global and local digital elevation or terrain-derived data.</li>
  <li><strong>UAV RGB and multispectral products</strong>, including very high-resolution orthomosaics, digital surface models, and vegetation-structure information.</li>
</ul>
</div>

<div class="floro-card">
<h3>Design motivation</h3>
<p>
The objective is not only to increase the number of pretraining samples, but to increase the diversity of sensing conditions encountered during representation learning.
</p>
<p>
FLORO therefore emphasizes variation across sensors, spatial resolutions, spectral definitions, terrain conditions, and ecological contexts.
</p>
</div>

</div>

<figure class="floro-figure">
  <img src="assets/images/pretraining_data.png" alt="FLORO pretraining data">
  <figcaption class="floro-caption">
    Overview of the heterogeneous pretraining data used to expose FLORO to diverse sensing conditions.
  </figcaption>
</figure>

</section>

<section class="floro-section">

<h2>🔨 Pretraining Regime</h2>

<div class="floro-panel">

<p class="justify">
FLORO is pretrained on approximately 80K remote sensing image samples spanning satellite, airborne, UAV, optical, SAR, and terrain-derived data sources. The model supports multispectral optical data, Sentinel-1 SAR backscatter, elevation products, UAV-derived structural information, and high-resolution imagery.
</p>

<p class="justify">
FLORO learns by reconstructing masked content from partially observed multimodal inputs, encouraging the encoder to capture both spectral and spatial structure. Availability and validity channels allow the model to distinguish between observed, missing, and invalid data, supporting transfer across datasets with different sensor configurations.
</p>

</div>

<figure class="floro-figure">
  <img src="assets/images/pretraining_regime.png" alt="FLORO pretraining regime">
  <figcaption class="floro-caption">
    Masked autoencoding pretraining regime used to learn transferable multimodal geospatial representations.
  </figcaption>
</figure>

</section>

<section class="floro-section">

<h2>🧪 Evaluation</h2>

<p class="justify">
FLORO is evaluated using the <strong>PANGAEA</strong> benchmark, a standardized framework for assessing the transferability of geospatial foundation models across diverse Earth observation tasks.
</p>

<p class="justify">
The evaluation follows a frozen-encoder protocol: the pretrained FLORO encoder remains fixed, while benchmark-defined downstream decoders are trained for each task. This setting isolates the quality and transferability of the learned representation, rather than measuring the performance ceiling obtainable through full end-to-end fine-tuning.
</p>

<div class="floro-highlight-grid">

<div class="floro-card">
<h3>Semantic segmentation</h3>
<p>Land-cover, flood, burn-scar, agricultural, and urban-mapping benchmarks.</p>
</div>

<div class="floro-card">
<h3>Scene classification</h3>
<p>Multispectral land-use and land-cover classification tasks.</p>
</div>

<div class="floro-card">
<h3>Regression</h3>
<p>Ecological and vegetation-structure tasks such as biomass estimation and canopy-height reconstruction.</p>
</div>

<div class="floro-card">
<h3>Transfer evaluation</h3>
<p>Frozen-encoder testing emphasizes the quality and generality of the learned representation.</p>
</div>

</div>

<p class="justify">
Across these evaluations, FLORO achieves strong transfer performance despite being pretrained on substantially fewer image samples than several recent large-scale geospatial foundation models.
</p>

</section>

<section class="floro-section">

<h2>📊 Why FLORO?</h2>

<div class="floro-panel">

<p class="justify">
Many geospatial foundation models scale primarily by increasing the size of the pretraining corpus. FLORO explores a complementary direction: learning from a smaller but highly heterogeneous collection of remote sensing observations.
</p>

<p class="justify">
This design is motivated by ecological remote sensing, where downstream tasks often require generalization across sensors, landscapes, resolutions, and data availability conditions. In these settings, robustness to heterogeneous inputs can be as important as raw pretraining scale.
</p>

</div>

<ul class="floro-list">
  <li>Sensor diversity</li>
  <li>Modality flexibility</li>
  <li>Ecological relevance</li>
  <li>Transferability under limited downstream adaptation</li>
  <li>Compatibility with standardized benchmark evaluation</li>
</ul>

</section>

<section class="floro-section">

<h2>🌱 Applications</h2>

<p class="justify">
FLORO is being developed for remote sensing applications where ecological structure and function need to be inferred across space and scale.
</p>

<div class="floro-highlight-grid">

<div class="floro-card">
<h3>Vegetation and land-cover mapping</h3>
<p>Mapping vegetation patterns and land-cover structure across heterogeneous landscapes.</p>
</div>

<div class="floro-card">
<h3>Canopy-height estimation</h3>
<p>Supporting vegetation-structure mapping from multimodal remote sensing observations.</p>
</div>

<div class="floro-card">
<h3>Aboveground biomass estimation</h3>
<p>Linking spectral, structural, and environmental signals to ecosystem function.</p>
</div>

<div class="floro-card">
<h3>Dryland and rangeland monitoring</h3>
<p>Supporting ecological monitoring in sparse, heterogeneous, and data-limited landscapes.</p>
</div>

<div class="floro-card">
<h3>Multimodal data fusion</h3>
<p>Combining optical, radar, elevation, and UAV-derived data sources.</p>
</div>

<div class="floro-card">
<h3>Limited-label transfer learning</h3>
<p>Adapting remote sensing representations to downstream datasets with limited annotations.</p>
</div>

</div>

</section>

<section class="floro-section">

<h2>📌 Status</h2>

<div class="floro-status">

<p class="justify">
The current version of FLORO has been evaluated under the PANGAEA benchmark using a frozen-encoder transfer protocol. The model has been tested across semantic segmentation, scene classification, and regression tasks, showing strong generalization across multiple datasets and sensing configurations.
</p>

<p class="justify">
Code, pretrained weights, and manuscript links will be added once publicly available.
</p>

</div>

</section>

</div>