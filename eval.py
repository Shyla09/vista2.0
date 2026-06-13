import os
import argparse
import torch
from torch.utils.data import DataLoader

from models.yolo_detector import YoloDetector
from models.clip_extractor import CLIPExtractor
from models.vista_model import VISTAModel
from utils.dataset import RefCOCOgDataset
from utils.loss import VISTALoss
from utils.engine import evaluate, EMA

def parse_args():
    parser = argparse.ArgumentParser(description="VISTA 2.0 Visual Grounding Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory of datasets")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="Pretrained CLIP model name")
    parser.add_argument("--use_synthetic", action="store_true", help="Force synthetic dataset mode")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA weights for evaluation")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")
    
    # 1. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at: {args.checkpoint}")
        
    print(f"Loading checkpoint from: {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint.get('args', argparse.Namespace())
    
    # Override settings using saved checkpoint arguments
    num_layers = getattr(train_args, 'num_layers', 3)
    num_heads = getattr(train_args, 'num_heads', 8)
    max_proposals = getattr(train_args, 'max_proposals', 20)
    clip_model_name = getattr(train_args, 'clip_model', args.clip_model)
    
    # 2. Setup Dataset & Dataloader
    print("\nSetting up evaluation dataset...")
    dataset = RefCOCOgDataset(
        data_root=args.data_root,
        split=args.split,
        use_synthetic=args.use_synthetic
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if dataset.use_synthetic else 4
    )
    
    # 3. Initialize Extractors
    print("\nInitializing Frozen Feature Extractors...")
    yolo_detector = YoloDetector(
        model_name="yolov8n.pt",
        max_proposals=max_proposals
    ).to(device)
    
    clip_extractor = CLIPExtractor(
        model_name=clip_model_name,
        project_dim=None
    ).to(device)
    
    embed_dim = clip_extractor.project_dim
    
    # 4. Initialize VISTA Grounding Model & Load Weights
    print(f"\nInitializing VISTA Core Model (embed_dim={embed_dim}) & Restoring Weights...")
    model = VISTAModel(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_relations=6
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("  - VISTA weights loaded successfully.")
    
    if clip_extractor.text_projection is not None and checkpoint.get('clip_proj_state_dict') is not None:
        clip_extractor.text_projection.load_state_dict(checkpoint['clip_proj_state_dict'])
        print("  - CLIP Text Projection weights loaded successfully.")
        
    # Recreate EMA helper to apply shadow weights
    ema = None
    if args.use_ema and checkpoint.get('ema_shadow') is not None:
        ema = EMA(model)
        # Manually load the shadow dict
        ema.shadow = {k: v.to(device) for k, v in checkpoint['ema_shadow'].items()}
        print("  - EMA shadow weights loaded and ready to apply.")
        
    # Loss criterion
    criterion = VISTALoss(
        lambda_rel=1.0,
        lambda_nce=1.0,
        lambda_box=5.0,
        temperature=0.07
    )
    
    # 5. Run Evaluation
    print("\nRunning Evaluation...")
    metrics = evaluate(
        model=model,
        yolo_detector=yolo_detector,
        clip_extractor=clip_extractor,
        criterion=criterion,
        dataloader=loader,
        device=device,
        ema=ema
    )
    
    # Print metrics
    print("\n==============================================")
    print(f"        EVALUATION RESULTS ({args.split.upper()} SPLIT)")
    print("==============================================")
    print(f"Acc@0.5:  {metrics['acc_0.5']*100:.2f}%")
    print(f"Acc@0.25: {metrics['acc_0.25']*100:.2f}%")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print("----------------------------------------------")
    print(f"Eval Total Loss:  {metrics['eval_loss_total']:.4f}")
    print(f"Eval CE Loss:     {metrics['eval_loss_ce']:.4f}")
    print(f"Eval Box GIoU:    {metrics['eval_loss_box']:.4f}")
    print(f"Eval NCE Loss:    {metrics['eval_loss_nce']:.4f}")
    print(f"Eval Relation:    {metrics['eval_loss_rel']:.4f}")
    print("==============================================")

if __name__ == '__main__':
    main()
