import os
import argparse
import random
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

from models.yolo_detector import YoloDetector
from models.clip_extractor import CLIPExtractor
from models.vista_model import VISTAModel
from utils.dataset import RefCOCOgDataset
from utils.loss import VISTALoss
from utils.engine import train_one_epoch, evaluate, EMA

def parse_args():
    parser = argparse.ArgumentParser(description="VISTA 2.0 Visual Grounding Training Script")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory of datasets")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for trainable weights")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of stacked geometric attention layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--max_proposals", type=int, default=20, help="Max number of candidate boxes from YOLO")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="Pretrained CLIP model name")
    parser.add_argument("--use_synthetic", action="store_true", help="Force synthetic dataset mode")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--coco_train_dir", type=str, default=None, help="Path to COCO train2014 images folder")
    parser.add_argument("--coco_val_dir", type=str, default=None, help="Path to COCO val2014 images folder")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of the dataset to train/validate on (0.0 to 1.0)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 1. Initialize Datasets and Dataloaders
    print("\nSetting up datasets...")
    train_dataset = RefCOCOgDataset(
        data_root=args.data_root, 
        split="train", 
        use_synthetic=args.use_synthetic,
        coco_train_dir=args.coco_train_dir,
        coco_val_dir=args.coco_val_dir
    )
    
    # If the training dataset fallback to synthetic, make validation fallback too
    use_synth_fallback = train_dataset.use_synthetic
    val_dataset = RefCOCOgDataset(
        data_root=args.data_root, 
        split="val", 
        use_synthetic=use_synth_fallback,
        coco_train_dir=args.coco_train_dir,
        coco_val_dir=args.coco_val_dir
    )
    
    # Apply subset fraction if less than 1.0
    if args.subset_fraction < 1.0:
        # For training
        num_train_samples = max(1, int(len(train_dataset) * args.subset_fraction))
        train_indices = random.sample(range(len(train_dataset)), num_train_samples)
        train_dataset = Subset(train_dataset, train_indices)
        print(f"Subsetting training dataset to {num_train_samples} samples (fraction={args.subset_fraction})")
        
        # For validation
        num_val_samples = max(1, int(len(val_dataset) * args.subset_fraction))
        val_indices = random.sample(range(len(val_dataset)), num_val_samples)
        val_dataset = Subset(val_dataset, val_indices)
        print(f"Subsetting validation dataset to {num_val_samples} samples (fraction={args.subset_fraction})")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0 if use_synth_fallback else 4, 
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0 if use_synth_fallback else 4
    )
    
    # 2. Initialize Frozen Extraction Models
    print("\nInitializing Frozen Feature Extractors...")
    yolo_detector = YoloDetector(
        model_name="yolov8n.pt", 
        max_proposals=args.max_proposals
    ).to(device)
    
    # Pass project_dim=None so it uses CLIPExtractor's auto-detected vision hidden size
    clip_extractor = CLIPExtractor(
        model_name=args.clip_model, 
        project_dim=None
    ).to(device)
    
    embed_dim = clip_extractor.project_dim
    
    # 3. Initialize VISTA Grounding Model
    print(f"\nInitializing VISTA Core Model (embed_dim={embed_dim})...")
    model = VISTAModel(
        embed_dim=embed_dim, 
        num_heads=args.num_heads, 
        num_layers=args.num_layers, 
        num_relations=6
    ).to(device)
    
    # Set up EMA
    ema = EMA(model, decay=0.999)
    
    # 4. Setup Loss Criterion and Optimizer
    criterion = VISTALoss(
        lambda_rel=1.0, 
        lambda_nce=1.0, 
        lambda_box=5.0, 
        temperature=0.07
    )
    
    # Optimizer (only train the VISTA parameters and the CLIP text projection weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if clip_extractor.text_projection is not None:
        trainable_params += [p for p in clip_extractor.text_projection.parameters() if p.requires_grad]
        
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine Annealing scheduler (T_0=5 epochs, T_mult=2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5, 
        T_mult=2, 
        eta_min=1e-6
    )
    
    # Mixed precision scaler (FP16)
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None
    
    # Resume logic
    start_epoch = 1
    best_acc = 0.0
    
    resume_path = args.resume
    if resume_path is None:
        latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
        if os.path.exists(latest_path):
            print(f"\nAuto-detecting latest checkpoint at {latest_path}. Setting as resume checkpoint.")
            resume_path = latest_path
            
    if resume_path is not None:
        if os.path.exists(resume_path):
            print(f"\nResuming training from checkpoint: {resume_path}...")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            if clip_extractor.text_projection is not None and checkpoint.get('clip_proj_state_dict') is not None:
                clip_extractor.text_projection.load_state_dict(checkpoint['clip_proj_state_dict'])
                
            # Load EMA shadow
            if checkpoint.get('ema_shadow') is not None:
                ema.shadow = {k: v.to(device) for k, v in checkpoint['ema_shadow'].items()}
                
            # Load optimizer and scheduler states
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                # Fallback: step scheduler up to start epoch
                for _ in range(checkpoint['epoch']):
                    scheduler.step()
                    
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('best_acc', 0.0)
            print(f"Resumed successfully at epoch {start_epoch} (best Acc@0.5: {best_acc*100:.2f}%)")
        else:
            print(f"\nWarning: Resume checkpoint '{resume_path}' not found. Starting training from scratch.")
            
    print("\nStarting Training Pipeline...")
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            yolo_detector=yolo_detector,
            clip_extractor=clip_extractor,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            ema=ema,
            max_grad_norm=1.0
        )
        
        # Step learning rate scheduler
        scheduler.step()
        
        # Evaluate on validation split
        val_metrics = evaluate(
            model=model,
            yolo_detector=yolo_detector,
            clip_extractor=clip_extractor,
            criterion=criterion,
            dataloader=val_loader,
            device=device,
            ema=ema
        )
        
        # Print metrics summaries
        print(f"Epoch {epoch} Training Metrics:")
        print(f"  - Total Loss: {train_metrics['loss_total']:.4f}")
        print(f"  - Grounding CE Loss: {train_metrics['loss_ce']:.4f}")
        print(f"  - Box Reg GIoU Loss: {train_metrics['loss_box']:.4f}")
        print(f"  - Contrastive InfoNCE Loss: {train_metrics['loss_nce']:.4f}")
        print(f"  - Aux Relation Loss: {train_metrics['loss_rel']:.4f}")
        
        print(f"Epoch {epoch} Validation Metrics:")
        print(f"  - Acc@0.5:  {val_metrics['acc_0.5']*100:.2f}%")
        print(f"  - Acc@0.25: {val_metrics['acc_0.25']*100:.2f}%")
        print(f"  - Mean IoU: {val_metrics['mean_iou']:.4f}")
        print(f"  - Val CE Loss:  {val_metrics['eval_loss_ce']:.4f}")
        
        # Save checkpoints
        is_best = val_metrics['acc_0.5'] > best_acc
        if is_best:
            best_acc = val_metrics['acc_0.5']
            print(f"  => New best validation Acc@0.5: {best_acc*100:.2f}%! Saving best checkpoint.")
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'clip_proj_state_dict': clip_extractor.text_projection.state_dict() if clip_extractor.text_projection is not None else None,
            'ema_shadow': ema.shadow,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'args': args
        }
        
        # Save current epoch checkpoint
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "checkpoint_best.pt"))
            
    print(f"\nTraining complete! Best validation Acc@0.5 achieved: {best_acc*100:.2f}%")

if __name__ == '__main__':
    main()
