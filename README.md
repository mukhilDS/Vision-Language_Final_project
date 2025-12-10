# Flickr30k Image–Text Retrieval Using BLIP: Pretrained Evaluation, Fine-Tuning, and Improved Projection Head

This repository contains an end-to-end Flickr30k image–text retrieval project using the BLIP (Bootstrapped Language–Image Pretraining) model. The project evaluates the pretrained BLIP retrieval model, performs baseline fine-tuning of the projection head, and introduces an improved multi-layer projection head that increases Recall@K performance. All experiments were executed in Google Colab (A100 GPU), and outputs are saved under the `outputs/` directory.

## Dataset

This project uses the HuggingFace parquet-based Flickr30k dataset:

```
lmms-lab/flickr30k
```

Each entry contains:
- `image`: PIL image  
- `caption`: a single ground-truth caption  
- `img_id`: numeric identifier  

The dataset is fully compatible with Python 3.12 and does not require any manual downloads.

## Model Architecture

The retrieval pipeline uses the BLIP retrieval model implemented in the LAVIS library.

### Pretrained Encoders
- Vision encoder: ViT-B/16  
- Text encoder: BERT-style transformer  
- Encoders remain frozen during all training  
- Retrieval operates on their output representations  

### Baseline Projection Head
A single linear projection layer maps image and text embeddings into a shared space:

```
image_features → Linear(d, d)
text_features → Linear(d, d)
```

### Improved Projection Head
A deeper MLP is introduced to improve cross-modal alignment:

```
input_dim → Linear → ReLU → Linear → output_dim
```

This improves representational power and retrieval accuracy.

## Training Procedure

### Baseline Fine-Tuning
- Optimizer: AdamW  
- Learning rate: 1e-3  
- Weight decay: 1e-4  
- Epochs: 5  
- Batch size: 512  
- Runtime: ~3 minutes on an A100 GPU  

### Improved Projection Head Training
- Same optimizer and learning rate  
- Epochs: 8  
- Runtime: ~3 minutes  

Only the projection layers are trained; all BLIP encoders remain frozen.

## Quantitative Results (Recall@K)

### Image → Text Retrieval

| Model              | R@1    | R@5    | R@10   |
|--------------------|--------|--------|--------|
| Pretrained BLIP    | 0.1214 | 0.2897 | 0.3776 |
| Baseline Fine-Tune | 0.1985 | 0.4021 | 0.5023 |
| Improved (MLP)     | 0.2527 | 0.4662 | 0.5616 |

### Text → Image Retrieval

| Model              | R@1    | R@5    | R@10   |
|--------------------|--------|--------|--------|
| Pretrained BLIP    | 0.1302 | 0.3045 | 0.3928 |
| Baseline Fine-Tune | 0.2149 | 0.4157 | 0.5164 |
| Improved (MLP)     | 0.2609 | 0.4808 | 0.5826 |

The improved projection head outperforms the pretrained model and the baseline fine-tuned model across all Recall@K metrics.

## Qualitative Results

### Image → Text Retrieval
For several test images, the evaluation compares:
- Ground-truth human captions  
- Pretrained BLIP top-5 retrieved captions  
- Baseline fine-tuning top-5 retrieved captions  
- Improved MLP head top-5 retrieved captions  

Observations:
- The pretrained model often retrieves generic or weakly relevant captions.  
- The baseline model improves relevance and consistency.  
- The improved head retrieves captions that more accurately capture scene details, objects, and relationships.

### Text → Image Retrieval
For sample queries such as “Several motorcycle policemen driving on a street”:
- The pretrained system returns unrelated images.  
- The baseline model retrieves partially relevant images.  
- The improved model retrieves the correct police/motorcycle street scene images reliably.

Qualitative examples appear in Section 8 of the notebook.

## Output Directory Structure

```
outputs/
├── pretrained_eval/
│   └── metrics.json
├── baseline_run/
│   └── metrics_finetuned.json
├── improved_run/
│   └── metrics_improved.json
├── logs/
├── checkpoints/
└── final_summary.json
```

Each metrics file stores Recall@1, Recall@5, and Recall@10 for both retrieval directions.

## Running the Project (Google Colab)

1. Open the notebook:
```
notebook/flickr30k_final_project.ipynb
```

2. Install required dependencies:
```bash
pip install lavis datasets pillow
```

3. Run all cells. The notebook will:
- Load Flickr30k  
- Load pretrained BLIP  
- Extract image/text embeddings  
- Train baseline projection head  
- Train improved projection head  
- Evaluate all models with Recall@K  
- Save outputs to `outputs/`  

The notebook is self-contained and fully reproducible.

## Summary

This project shows that:
- Fine-tuning even a shallow projection head improves retrieval performance over the pretrained BLIP model.  
- A deeper projection MLP yields additional gains and achieves the highest recall across all metrics.  
- Both quantitative and qualitative results confirm the effectiveness of extending BLIP with improved alignment layers.

## References

- Li et al., “BLIP: Bootstrapped Language-Image Pretraining.”  
- Young et al., “Flickr30k Entities.”  
- HuggingFace Datasets Documentation.  
- LAVIS Library Documentation.  
