# Vision & Language Final Project  
## Flickr30k Image–Text Retrieval with BLIP-2

This repository contains the full code, experiments, and report artifacts for my **CSE 589 (Vision & Language)** course project.

The project follows the official requirements:

1. Select a **vision–language task**  
2. Identify the **State-of-the-Art (SOTA)** model for that task  
3. **Replicate** the method using publicly available code/checkpoints  
4. Implement **at least one improvement** and compare results  
5. Produce a final report following the official course template  

---

# 1. Project Overview

- **Task:** Image–Text Retrieval (Image→Text and Text→Image)  
- **Model:** **BLIP-2** (`Salesforce/blip2-flan-t5-xl`, via HuggingFace Transformers)  
- **Dataset:** **Flickr30k** (HuggingFace `lmms-lab/flickr30k` Parquet release)  

The goal is to build an end-to-end pipeline for:

- **Image→Text (I2T):** Given an image, retrieve the correct caption.  
- **Text→Image (T2I):** Given a caption, retrieve the correct image.  

We treat BLIP-2 as the **SOTA model** for Flickr30k retrieval, and we:

1. Evaluate the **pretrained BLIP-2** checkpoint (no Flickr30k training).  
2. Train a **baseline projection head** on a 5k subset of Flickr30k.  
3. Train an **improved 2-layer MLP projection head** on the same subset.  


The final **report PDF** and **Google Colab notebook** are included in this repo.

---

#  2. Report Structure

The final report strictly follows the official  
**“05\_Course Project Report Template”**:

1. **Task Description**  
2. **Related Work & SOTA Identification**  
3. **Approach (Model Details)**  
4. **Dataset(s)**  
5. **Experimental Results**  
6. **Possible Improvements & Results**  
7. **Code Repository Link (this repo)**  
8. **References**

The compiled PDF (e.g., `V_L_Final_Report.pdf`) is in the `report/` folder.

---

# 3. Related Work (Summary)

We focus on **image–text retrieval** and compare several major vision–language models:

| Paper / Model         | Task                    | Key Idea                                                 | Flickr30k I2T R@1 (reported) |
|-----------------------|-------------------------|----------------------------------------------------------|------------------------------|
| **CLIP (2021)**       | Generic VL retrieval    | Dual encoders trained on 400M image–text pairs           | 88.0%                        |
| **ALBEF (2021)**      | VL retrieval            | Align-Before-Fuse with momentum distillation             | 95.9%                        |
| **BLIP (2022)**       | Captioning + retrieval  | Bootstrapped language–image pretraining                  | 96.1%                        |
| **X-VLM (2022)**      | Unified VL pretraining  | Single model across detection, captioning, retrieval     | 96.4%                        |
| **BLIP-2 (2023)**     | VL + LLM integration    | Q-Former bridging frozen vision encoder + LLM            | **96.7%**                    |

### 🔎 Why BLIP-2 as SOTA?

BLIP-2 outperforms or matches previous models on Flickr30k and MS-COCO retrieval, while using a clean architecture:

- Frozen **vision encoder** + **T5** language model  
- A **Q-Former** that learns to connect vision and language  
- Strong results with relatively lightweight fine-tuning  

For this project, BLIP-2 offers:

- Clear SOTA status for retrieval  
- Stable, public HF implementation  
- Enough structure to design and test improvements (projection head changes)

---

#  4. Experiments

All experiments are run from a **single Colab notebook**:

> `notebook/flickr30k_final_project.ipynb`

Environment (for reproducibility):

- **Platform:** Google Colab Pro  
- **GPU:** A100 (80 GB)  
- **RAM:** High RAM setting  
- **Core Libraries:**
  - `torch`
  - `transformers`
  - `datasets`
  - `accelerate`
  - `numpy`, `scikit-learn`, `matplotlib`

We evaluate three model variants:

1. **Pretrained:** BLIP-2 checkpoint, used as-is.  
2. **Baseline:** BLIP-2 with a **linear** projection head, trained on **5k** Flickr30k pairs.  
3. **Improved:** BLIP-2 with a **2-layer MLP** projection head, trained on the same 5k pairs with a slightly longer schedule.

All results are reported as **Recall@K (%)** for both I2T and T2I.

---

## 4.1 SOTA Replication (Released Weights)

We first evaluate the **official pretrained BLIP-2 checkpoint** without any Flickr30k training of the projection heads.

| Model        | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | Notes                                   |
|-------------|---------|---------|----------|---------|---------|----------|-----------------------------------------|
| BLIP-2 (paper, full recipe) | 96.7 | –       | 99.9     | 95.0+   | –       | 99.0+    | SOTA (full training)                    |
| **Pretrained (ours)**       | **0.00** | **0.01** | **0.03**  | **0.00** | **0.02** | **0.04**  | No Flickr30k-specific alignment layers |

**Interpretation:**  
The generic BLIP-2 checkpoint alone is **not** sufficient for Flickr30k retrieval in this setup.
Without task-specific alignment (projection head training), Recall@K is essentially zero.
This motivates our fine-tuning experiments.

---

## 4.2 Training Replication (Our Training Run)

We then train BLIP-2 projection heads on a **5k subset** of the Flickr30k training split:

- Encoders + Q-Former are **frozen**  
- Only projection heads are trained  
- Full test split is used for evaluation  

| Model             | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | Explanation                                          |
|------------------|---------|---------|----------|---------|---------|----------|------------------------------------------------------|
| BLIP-2 (paper)   | 96.7    | –       | 99.9     | 95.0+   | –       | 99.0+    | Full recipe, end-to-end training                     |
| **Baseline (5k)**| **20.67** | **41.17** | **51.10**  | **20.07** | **40.47** | **50.96**  | Frozen backbone + linear projection head on 5k pairs |

**Interpretation:**  
- Training only small projection heads already lifts R@10 from ~0% → **~51%**.  
- This confirms that **BLIP-2 needs alignment layers** tuned to Flickr30k to work well in retrieval.  

---

## 4.3 Improvement Attempt

We propose a single, clear improvement:

> 🔧 Replace the linear projection head with a **2-layer MLP**  
> (Linear → GELU → Linear), same output dimension (768).

Training setup:

- Same 5k training subset  
- Slightly more epochs (e.g., 8 vs 5)  
- Tuned learning rate for stability  

###  Baseline vs Improved

| Model                    | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|--------------------------|---------|---------|----------|---------|---------|----------|
| **Baseline (linear head)**   | 20.67  | 41.17  | 51.10   | 20.07  | 40.47  | 50.96   |
| **Improved (2-layer MLP)**   | **25.27**  | **46.62**  | **56.16**   | **26.10**  | **48.08**  | **58.26**   |

###  Performance Discussion

- The **pretrained** model is near random on Flickr30k in this setup.  
- The **baseline** fine-tuned projection head already gives a **huge jump** in performance.  
- The **improved MLP head** adds **another ~4–6 percentage points** across all R@K metrics.  

The PCA plots (in the report) show that image and text embeddings become:

- **More compact**  
- **More overlapping**  
- **Better aligned across modalities**  

This visually matches the numerical improvements.

---

# 5. Repository Contents

```text
Vision-Language_Final_project/
├── notebook/
│   └── flickr30k_final_project.ipynb   # Main Colab notebook (full pipeline)
│
├── outputs/
│   ├── pretrained_eval/
│   │   └── metrics.json                # Pretrained BLIP-2 evaluation
│   ├── baseline_run/
│   │   └── metrics_finetuned.json      # Baseline linear head results
│   └── improved_run/
│       └── metrics_improved.json       # Improved MLP head results
│
├── figures/
│   ├── fig_recall_comparison.png       # Recall@K bar plot
│   ├── fig_pca_pretrained.png          # PCA of embeddings (pretrained)
│   ├── fig_pca_trained.png             # PCA of embeddings (improved model)
│   ├── fig_sample_image_24132.png      # Sample Flickr30k image (qualitative)
│   ├── fig_retrieved_set1.png          # Retrieved images – example set 1
│   ├── fig_retrieved_set2.png          # Retrieved images – example set 2
│   └── fig_retrieved_set3.png          # Retrieved images – example set 3
│
├── report/
│   ├── main.tex                        # LaTeX source following course template
│   └── V_L_Final_Report.pdf            # Final compiled report
│
└── README.md                           # This file
