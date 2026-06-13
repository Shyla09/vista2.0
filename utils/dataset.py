import os
import pickle
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
import random
from torchvision import transforms

class RefCOCOgDataset(Dataset):
    """
    RefCOCOg Dataset (UMD Split) for VISTA 2.0.
    Loads images and referring expressions, parses relations, and supports
    hard negative mining. Automatically falls back to synthetic/mock data
    if dataset files are not found.
    """
    def __init__(self, data_root="data", split="train", transform=None, use_synthetic=False,
                 coco_train_dir=None, coco_val_dir=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.use_synthetic = use_synthetic
        
        self.ref_path = os.path.join(data_root, "refcocog", f"refs(umd).p")
        if not os.path.exists(self.ref_path):
            self.ref_path = os.path.join(data_root, f"refs(umd).p")
            
        self.coco_instances_path = os.path.join(data_root, "refcocog", "instances.json")
        if not os.path.exists(self.coco_instances_path):
            self.coco_instances_path = os.path.join(data_root, "instances.json")
        
        # Resolve train images directory (handles standard, custom, and nested paths)
        if coco_train_dir is not None:
            self.coco_train_img_dir = coco_train_dir
        else:
            base_train = os.path.join(data_root, "coco", "train2014")
            nested_train = os.path.join(base_train, "train2014")
            self.coco_train_img_dir = nested_train if os.path.exists(nested_train) and os.path.isdir(nested_train) else base_train
            
        # Resolve val images directory (handles standard, custom, and nested paths)
        if coco_val_dir is not None:
            self.coco_val_img_dir = coco_val_dir
        else:
            base_val = os.path.join(data_root, "coco", "val2014")
            nested_val = os.path.join(base_val, "val2014")
            self.coco_val_img_dir = nested_val if os.path.exists(nested_val) and os.path.isdir(nested_val) else base_val
        
        self.items = []
        self.ann_to_bbox = {}
        self.image_id_to_filename = {}
        self.image_to_annotations = {}
        
        real_files_exist = (
            os.path.exists(self.ref_path) and 
            os.path.exists(self.coco_instances_path)
        )
        
        if real_files_exist and not self.use_synthetic:
            print(f"Loading RefCOCOg real dataset ({split} split)...")
            self._load_real_dataset()
        else:
            print(f"Dataset files not found or synthetic mode enabled. Initializing SYNTHETIC dataset for {split} split...")
            self.use_synthetic = True
            self._init_synthetic_dataset()
            
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                     (0.26862954, 0.26130258, 0.27577711))
            ])

    def _load_real_dataset(self):
        with open(self.coco_instances_path, 'r') as f:
            coco_data = json.load(f)
            
        for img in coco_data['images']:
            self.image_id_to_filename[img['id']] = img['file_name']
            
        for ann in coco_data['annotations']:
            ann_id = ann['id']
            x, y, w, h = ann['bbox']
            bbox_xyxy = [x, y, x + w, y + h]
            
            self.ann_to_bbox[ann_id] = {
                'bbox': bbox_xyxy,
                'category_id': ann['category_id'],
                'image_id': ann['image_id']
            }
            
            image_id = ann['image_id']
            if image_id not in self.image_to_annotations:
                self.image_to_annotations[image_id] = []
            self.image_to_annotations[image_id].append(ann)

        with open(self.ref_path, 'rb') as f:
            refs = pickle.load(f)
            
        for ref in refs:
            if ref['split'] != self.split:
                continue
                
            ann_id = ref['ann_id']
            image_id = ref['image_id']
            category_id = ref['category_id']
            
            if ann_id not in self.ann_to_bbox or image_id not in self.image_id_to_filename:
                continue
                
            gt_bbox_abs = self.ann_to_bbox[ann_id]['bbox']
            
            distractor_ann_ids = []
            for ann in self.image_to_annotations.get(image_id, []):
                if ann['category_id'] == category_id and ann['id'] != ann_id:
                    distractor_ann_ids.append(ann['id'])
            
            for sent in ref['sentences']:
                query_text = sent['sent']
                relation_id = self.parse_relation(query_text)
                
                self.items.append({
                    'ref_id': ref['ref_id'],
                    'image_id': image_id,
                    'ann_id': ann_id,
                    'file_name': self.image_id_to_filename[image_id],
                    'query': query_text,
                    'gt_bbox': gt_bbox_abs,
                    'category_id': category_id,
                    'relation_id': relation_id,
                    'distractor_ann_ids': distractor_ann_ids
                })
                
        print(f"Loaded {len(self.items)} samples from real dataset split: {self.split}")

    def _init_synthetic_dataset(self, num_samples=100):
        categories = {1: "cup", 2: "laptop", 3: "box", 4: "chair", 5: "person"}
        relations_vocab = [
            ("to the left of", 0), ("on the left of", 0),
            ("to the right of", 1), ("on the right of", 1),
            ("above", 2), ("on top of", 2),
            ("below", 3), ("under", 3),
            ("next to", 4), ("near", 4), ("beside", 4)
        ]
        
        for idx in range(num_samples):
            target_cat_id = random.choice(list(categories.keys()))
            landmark_cat_id = random.choice(list(categories.keys()))
            target_name = categories[target_cat_id]
            landmark_name = categories[landmark_cat_id]
            
            rel_phrase, rel_id = random.choice(relations_vocab)
            query = f"the {target_name} {rel_phrase} the {landmark_name}"
            
            if rel_id == 0:
                target_box = [0.1, 0.3, 0.3, 0.7]
                landmark_box = [0.6, 0.3, 0.8, 0.7]
            elif rel_id == 1:
                target_box = [0.6, 0.3, 0.8, 0.7]
                landmark_box = [0.1, 0.3, 0.3, 0.7]
            elif rel_id == 2:
                target_box = [0.35, 0.1, 0.65, 0.4]
                landmark_box = [0.35, 0.6, 0.65, 0.9]
            elif rel_id == 3:
                target_box = [0.35, 0.6, 0.65, 0.9]
                landmark_box = [0.35, 0.1, 0.65, 0.4]
            else:
                target_box = [0.2, 0.3, 0.4, 0.7]
                landmark_box = [0.5, 0.3, 0.7, 0.7]
                
            distractor_box = [0.8, 0.1, 0.95, 0.3] if rel_id != 1 else [0.05, 0.1, 0.2, 0.3]
            
            scale = 224.0
            gt_bbox_abs = [coord * scale for coord in target_box]
            distractor_bbox_abs = [coord * scale for coord in distractor_box]
            landmark_bbox_abs = [coord * scale for coord in landmark_box]
            
            self.items.append({
                'ref_id': f"synth_{idx}",
                'image_id': f"img_{idx}",
                'ann_id': f"ann_target_{idx}",
                'query': query,
                'gt_bbox': gt_bbox_abs,
                'category_id': target_cat_id,
                'relation_id': rel_id,
                'synthetic_boxes': {
                    'target': gt_bbox_abs,
                    'landmark': landmark_bbox_abs,
                    'distractor': distractor_bbox_abs
                }
            })

    def parse_relation(self, sentence):
        """
        Improvement #4: Simple Rule-Based Relation Parser (Whole-Word).
        """
        cleaned = sentence.lower()
        for char in [".", ",", ";", ":", "!", "?", "-", "_", "(", ")", "[", "]", "{", "}", "/"]:
            cleaned = cleaned.replace(char, " ")
        padded = f" {cleaned} "
        
        if " left " in padded:
            return 0
        elif " right " in padded:
            return 1
        elif any(f" {k} " in padded for k in ["above", "top", "upper", "over"]):
            return 2
        elif any(f" {k} " in padded for k in ["below", "bottom", "lower", "under"]):
            return 3
        elif any(f" {k} " in padded for k in ["next to", "near", "beside", "close", "by"]):
            return 4
        else:
            return 5

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # 1. Load image
        if self.use_synthetic:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            draw = ImageDraw.Draw(img)
            
            t_box = item['synthetic_boxes']['target']
            draw.rectangle(t_box, fill=(200, 50, 50), outline=(255, 0, 0))
            
            l_box = item['synthetic_boxes']['landmark']
            draw.rectangle(l_box, fill=(50, 50, 200), outline=(0, 0, 255))
            
            d_box = item['synthetic_boxes']['distractor']
            draw.rectangle(d_box, fill=(220, 100, 50), outline=(255, 128, 0))
            
            width, height = 224, 224
        else:
            if "train2014" in item['file_name']:
                img_path = os.path.join(self.coco_train_img_dir, item['file_name'])
            elif "val2014" in item['file_name']:
                img_path = os.path.join(self.coco_val_img_dir, item['file_name'])
            else:
                train_path = os.path.join(self.coco_train_img_dir, item['file_name'])
                img_path = train_path if os.path.exists(train_path) else os.path.join(self.coco_val_img_dir, item['file_name'])
                
            try:
                img = Image.open(img_path).convert("RGB")
                width, height = img.size
            except Exception as e:
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
                width, height = 224, 224
                
        gt_bbox_abs = item['gt_bbox']
        gt_bbox_norm = [
            gt_bbox_abs[0] / width,
            gt_bbox_abs[1] / height,
            gt_bbox_abs[2] / width,
            gt_bbox_abs[3] / height
        ]
        gt_bbox_norm = [max(0.0, min(1.0, coord)) for coord in gt_bbox_norm]
        
        distractors_norm = []
        if self.use_synthetic:
            d_box_abs = item['synthetic_boxes']['distractor']
            d_box_norm = [d_box_abs[0]/224, d_box_abs[1]/224, d_box_abs[2]/224, d_box_abs[3]/224]
            distractors_norm.append(d_box_norm)
        else:
            for ann_id in item['distractor_ann_ids']:
                if ann_id in self.ann_to_bbox:
                    d_box_abs = self.ann_to_bbox[ann_id]['bbox']
                    d_box_norm = [
                        d_box_abs[0] / width,
                        d_box_abs[1] / height,
                        d_box_abs[2] / width,
                        d_box_abs[3] / height
                    ]
                    d_box_norm = [max(0.0, min(1.0, c)) for c in d_box_norm]
                    distractors_norm.append(d_box_norm)
                    
        max_distractors = 5
        distractors_norm = distractors_norm[:max_distractors]
        distractor_mask = [1.0] * len(distractors_norm) + [0.0] * (max_distractors - len(distractors_norm))
        while len(distractors_norm) < max_distractors:
            distractors_norm.append([0.0, 0.0, 0.0, 0.0])
            
        img_tensor = self.transform(img)
        
        # Raw unnormalized tensor in [0, 1] range for YOLOv8
        raw_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img_raw_tensor = raw_transform(img)
        
        return {
            'image': img_tensor,
            'image_raw': img_raw_tensor,
            'query': item['query'],
            'gt_box': torch.tensor(gt_bbox_norm, dtype=torch.float32),
            'relation_id': torch.tensor(item['relation_id'], dtype=torch.long),
            'hard_negatives': torch.tensor(distractors_norm, dtype=torch.float32),
            'hard_negatives_mask': torch.tensor(distractor_mask, dtype=torch.float32),
            'image_size': torch.tensor([width, height], dtype=torch.float32)
        }
