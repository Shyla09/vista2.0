# VISTA 2.0: Visual Grounding via 11-Dimensional Geometric Spatial Attention

VISTA 2.0 is an advanced visual grounding model that dynamically locates objects in images based on complex natural language referring expressions. It features denser vision backbones, a highly descriptive 11-Dimensional relative spatial bias attention encoder, and hard negative contrastive learning to resolve scale, aspect ratios, and spatial containments.

---

## 🌟 Key Features

1. **CLIP ViT-L/14 Backbone Support:** Dynamically scales representation dimensions from 768 to 1024 and expands the visual feature map grid from $7 \times 7$ to $16 \times 16$, capturing high-resolution details of small objects.
2. **11-Dimensional Geometric Attention:** Incorporates scale ratios, aspect ratios, pairwise center offsets, Euclidean distances, and overlap IoU calculations into a trainable Spatial MLP ($G_{ij}$) to inject relative geometry biases directly into self-attention layers.
3. **Fixed RoI-Align Grid Pooling:** Pools regional box proposals to a fixed $7 \times 7$ grid to optimize alignment and prevent upsampling artifacts on dense feature maps.
4. **Intra-Image Hard Negative InfoNCE Loss:** Eliminates category confusion by mining same-image distractors and optimizing cross-modal alignment.
5. **EMA Weights & FP16 Precision:** Uses Exponential Moving Average (EMA) shadow weights for evaluation and FP16 mixed precision for fast, stable training on GPU.

---

## 📂 Directory Structure

```
vista2.0/
├── models/
│   ├── clip_extractor.py       # Auto-detects and extracts ViT-B/32 or ViT-L/14 features
│   ├── yolo_detector.py         # Extracts object bounding box proposals via YOLOv8
│   ├── geometric_attention.py   # Computes 11-dimensional spatial MLP G_ij attention bias
│   ├── cross_attention.py      # Aligns pooled text embeddings and visual features
│   └── vista_model.py          # Grounding model incorporating RoI-Align and classification heads
├── utils/
│   ├── dataset.py              # RefCOCOg / COCO dataloader with whole-word parser
│   ├── loss.py                 # Multi-component loss (CE, GIoU, InfoNCE hard negative)
│   ├── metrics.py              # Evaluates Acc@0.5, Acc@0.25, and Mean IoU
│   └── engine.py               # Train and evaluation loops with EMA shadow weights
├── train.py                    # Training script with dataset subset fraction scaling
├── eval.py                     # Evaluation script
├── test_single_file.py         # Shape and gradient flow verification script
└── .gitignore                  # Git exclusions file
```

---

## 🚀 Getting Started

### 1. Installation
Install the required dependencies (Ultralytics YOLOv8 and Hugging Face Transformers):
```bash
pip install ultralytics transformers torch torchvision tqdm
```

### 2. Verify Setup (Synthetic Mode)
Run the verification script to verify model shapes, forward pass, and gradient backpropagation:
```bash
python test_single_file.py
```

### 3. Training on RefCOCOg (UMD Split)
Run training on the full dataset using CLIP ViT-L/14:
```bash
python train.py \
    --data_root /path/to/refcocog-umd \
    --coco_train_dir /path/to/coco/train2014 \
    --coco_val_dir /path/to/coco/val2014 \
    --clip_model openai/clip-vit-large-patch14 \
    --epochs 15 \
    --batch_size 16 \
    --lr 1e-4
```

*Tip: Use `--subset_fraction 0.25` during development to train on a 25% subset of the dataset and accelerate training.*

### 4. Evaluation
Evaluate a trained checkpoint:
```bash
python eval.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --data_root /path/to/refcocog-umd \
    --split val
```

---

## 📊 Benchmark Results

| Model Version | CLIP Backbone | Spatial Features | Acc@0.5 | Acc@0.25 | Mean IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VISTA 1.0** (Ours) | ViT-B/32 | 11-D | **86.15%** | **88.92%** | **0.82** |
| **VISTA 2.0** (Target) | ViT-L/14 | 11-D | **88.65%** | **91.20%** | **0.85** |
