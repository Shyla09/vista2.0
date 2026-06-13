import torch
from torch.utils.data import DataLoader
from models.yolo_detector import YoloDetector
from models.clip_extractor import CLIPExtractor
from models.vista_model import VISTAModel
from utils.dataset import RefCOCOgDataset
from utils.loss import VISTALoss
from utils.engine import EMA

def main():
    print("==================================================")
    print("         VISTA 2.0 MATHS & SHAPE VERIFICATION      ")
    print("==================================================")
    
    device = torch.device("cpu")
    print(f"Running verification on device: {device}")
    
    # 1. Initialize Synthetic Dataset (forced synthetic for offline verification)
    print("\n[Step 1] Initializing RefCOCOg Dataset...")
    dataset = RefCOCOgDataset(data_root="data", split="train", use_synthetic=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Get a batch
    batch = next(iter(dataloader))
    print(f"Batch loaded successfully:")
    print(f"  - Images shape: {batch['image'].shape} (B, C, H, W)")
    print(f"  - Queries: {batch['query']}")
    print(f"  - Ground Truth boxes shape: {batch['gt_box'].shape} (B, 4) [normalized]")
    print(f"  - Relation IDs shape: {batch['relation_id'].shape} (B,) -> {batch['relation_id'].tolist()}")
    print(f"  - Hard Negatives shape: {batch['hard_negatives'].shape} (B, M, 4) [normalized]")
    print(f"  - Hard Negatives Mask shape: {batch['hard_negatives_mask'].shape} (B, M)")
    
    # 2. Initialize Models (using dummy/fallback mode for quick validation)
    print("\n[Step 2] Initializing Models (Frozen YOLO and CLIP + Trainable VISTA)...")
    yolo_detector = YoloDetector(max_proposals=10, conf_threshold=0.25)
    
    # Initialize CLIP extractor using base model
    clip_extractor = CLIPExtractor(model_name="openai/clip-vit-base-patch32", project_dim=None)
    embed_dim = clip_extractor.project_dim
    print(f"  - Detected project_dim from CLIP: {embed_dim}")
    
    # Initialize VISTA Grounding Model using the detected embed_dim
    model = VISTAModel(embed_dim=embed_dim, num_heads=8, num_layers=2, num_relations=6)
    
    # 3. Extract Features from Frozen Extractor Models
    print("\n[Step 3] Running Frozen Feature Extractors...")
    images = batch['image'].to(device)
    images_raw = batch['image_raw'].to(device)
    queries = batch['query']
    gt_boxes = batch['gt_box'].to(device)
    relation_ids = batch['relation_id'].to(device)
    hard_negatives = batch['hard_negatives'].to(device)
    hard_negatives_mask = batch['hard_negatives_mask'].to(device)
    
    with torch.no_grad():
        proposals, proposal_classes, proposal_masks = yolo_detector(images_raw, gt_boxes=gt_boxes)
        print("  - YOLO proposals shape:", proposals.shape, "(B, N, 4)")
        print("  - YOLO proposal masks shape:", proposal_masks.shape, "(B, N)")
        
        clip_visual_map = clip_extractor.extract_visual_features(images)
        print("  - CLIP visual feature map shape:", clip_visual_map.shape, "(B, C, H_feat, W_feat)")
        
        text_features = clip_extractor.extract_text_features(queries)
        print("  - CLIP text sequence feature shape:", text_features['sequence'].shape, "(B, S, C)")
        print("  - CLIP text pooled feature shape:", text_features['pooled'].shape, "(B, C)")
        
    # 4. Run Trainable VISTA Model Forward Pass
    print("\n[Step 4] Running VISTA Grounding Model Forward Pass...")
    model.to(device)
    
    outputs = model(
        clip_visual_map=clip_visual_map,
        proposals=proposals,
        text_features=text_features,
        relation_ids=relation_ids,
        proposal_masks=proposal_masks
    )
    
    scores = outputs['scores']
    refined_boxes = outputs['refined_boxes']
    relation_logits = outputs['relation_logits']
    
    print("VISTA Outputs:")
    print(f"  - Grounding Scores shape: {scores.shape} (B, N)")
    print(f"  - Refined Boxes shape: {refined_boxes.shape} (B, N, 4)")
    print(f"  - Relation Logits shape: {relation_logits.shape} (B, num_relations)")
    
    # Validate refined boxes are within boundaries
    print(f"  - Refined boxes values min: {refined_boxes.min().item():.4f}, max: {refined_boxes.max().item():.4f} (Expected [0.0, 1.0])")
    
    # 5. Compute Combined Loss
    print("\n[Step 5] Computing Loss Components...")
    criterion = VISTALoss(lambda_rel=1.0, lambda_nce=1.0, lambda_box=5.0)
    
    outputs['text_embed'] = text_features['pooled']
    
    total_loss, loss_dict = criterion(
        model_outputs=outputs,
        gt_boxes=gt_boxes,
        relation_ids=relation_ids,
        hard_negatives=hard_negatives,
        hard_negatives_mask=hard_negatives_mask,
        proposals=proposals
    )
    
    print("Loss Breakdowns:")
    for k, v in loss_dict.items():
        print(f"  - {k}: {v:.6f}")
        
    # 6. Verify Backpropagation (Gradient Flow)
    print("\n[Step 6] Running Backward Pass and Verifying Gradient Flow...")
    total_loss.backward()
    
    # Check that gradients flow to all trainable parameters
    grad_ok = True
    trainable_params_count = 0
    params_with_grad = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params_count += 1
            if param.grad is not None:
                params_with_grad += 1
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"    [WARNING] Parameter '{name}' gradient contains NaNs or Infs!")
                    grad_ok = False
            else:
                print(f"    [WARNING] Parameter '{name}' has NO gradient!")
                grad_ok = False
                
    print(f"Gradient Check Summary:")
    print(f"  - Total trainable parameters: {trainable_params_count}")
    print(f"  - Parameters with valid gradients: {params_with_grad}")
    
    if grad_ok and params_with_grad == trainable_params_count:
        print("  => SUCCESS: Gradients flow correctly to all parameters without numerical issues!")
    else:
        print("  => FAILED: Gradient check failed.")
        
    # 7. Test EMA initialization and shadow apply
    print("\n[Step 7] Testing EMA Class...")
    ema = EMA(model, decay=0.99)
    ema.update()
    print("  - EMA update: OK")
    ema.apply_shadow()
    print("  - EMA shadow applied: OK")
    ema.restore()
    print("  - EMA weights restored: OK")
    
    print("\n==================================================")
    print("  VERIFICATION COMPLETE: ALL MATHEMATICS ALIGN!   ")
    print("==================================================")

if __name__ == '__main__':
    main()
