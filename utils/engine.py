import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import evaluate_grounding

class EMA:
    """
    Exponential Moving Average for model weights.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Keep track of shadow weights on the correct device
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Updates the shadow weights with current parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # shadow = decay * shadow + (1 - decay) * param
                self.shadow[name] = (self.decay * self.shadow[name] + 
                                     (1.0 - self.decay) * param.data).clone()

    def apply_shadow(self):
        """
        Replaces current model weights with shadow weights (for evaluation).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """
        Restores original weights (to continue training).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}


def train_one_epoch(model, yolo_detector, clip_extractor, criterion, dataloader, optimizer, 
                    device, scaler=None, ema=None, max_grad_norm=1.0):
    model.train()
    total_loss_dict = {}
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        # Move data tensors to device
        images = batch['image'].to(device)
        images_raw = batch['image_raw'].to(device)
        queries = batch['query']
        gt_boxes = batch['gt_box'].to(device)
        relation_ids = batch['relation_id'].to(device)
        hard_negatives = batch['hard_negatives'].to(device)
        hard_negatives_mask = batch['hard_negatives_mask'].to(device)
        
        # 1. Feature Extraction (No Gradients for Frozen Networks)
        with torch.no_grad():
            # Proposals: (B, N, 4), labels: (B, N), masks: (B, N)
            # Use raw unnormalized images for YOLOv8
            proposals, _, proposal_masks = yolo_detector(images_raw, gt_boxes=gt_boxes)
            
            # CLIP Vision Feature Map: (B, embed_dim, H_feat, W_feat)
            # Use CLIP normalized images
            clip_visual_map = clip_extractor.extract_visual_features(images)
            
            # CLIP Text Features: dict with 'sequence' (B, S, embed_dim) and 'mask' (B, S)
            text_features = clip_extractor.extract_text_features(queries)
            
        # 2. Main Model Forward & Loss Computation
        optimizer.zero_grad()
        
        # Use mixed precision if scaler is provided
        if scaler is not None and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(
                    clip_visual_map=clip_visual_map,
                    proposals=proposals,
                    text_features=text_features,
                    relation_ids=relation_ids,
                    proposal_masks=proposal_masks
                )
                loss, loss_dict = criterion(
                    model_outputs=outputs,
                    gt_boxes=gt_boxes,
                    relation_ids=relation_ids,
                    hard_negatives=hard_negatives,
                    hard_negatives_mask=hard_negatives_mask,
                    proposals=proposals
                )
            
            # Scaled Backward & Optimizer Step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard Precision (CPU fallback)
            outputs = model(
                clip_visual_map=clip_visual_map,
                proposals=proposals,
                text_features=text_features,
                relation_ids=relation_ids,
                proposal_masks=proposal_masks
            )
            loss, loss_dict = criterion(
                model_outputs=outputs,
                gt_boxes=gt_boxes,
                relation_ids=relation_ids,
                hard_negatives=hard_negatives,
                hard_negatives_mask=hard_negatives_mask,
                proposals=proposals
            )
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            
        # Update EMA if available
        if ema is not None:
            ema.update()
            
        # Accumulate loss metrics for print/logging
        for k, v in loss_dict.items():
            total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v
            
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': loss.item(),
            'IoU': loss_dict['mean_target_iou']
        })
        
    # Average out metrics
    num_batches = len(dataloader)
    epoch_metrics = {k: v / num_batches for k, v in total_loss_dict.items()}
    
    return epoch_metrics


def evaluate(model, yolo_detector, clip_extractor, criterion, dataloader, device, ema=None):
    model.eval()
    
    # Temporarily apply EMA weights if available
    if ema is not None:
        ema.apply_shadow()
        
    total_loss_dict = {}
    all_pred_boxes = []
    all_gt_boxes = []
    all_pred_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            images_raw = batch['image_raw'].to(device)
            queries = batch['query']
            gt_boxes = batch['gt_box'].to(device)
            relation_ids = batch['relation_id'].to(device)
            hard_negatives = batch['hard_negatives'].to(device)
            hard_negatives_mask = batch['hard_negatives_mask'].to(device)
            
            # Extract features from frozen networks
            # During evaluation, we also include GT box in proposals to test upper-bound grounding
            proposals, _, proposal_masks = yolo_detector(images_raw, gt_boxes=gt_boxes)
            clip_visual_map = clip_extractor.extract_visual_features(images)
            text_features = clip_extractor.extract_text_features(queries)
            
            # Forward
            outputs = model(
                clip_visual_map=clip_visual_map,
                proposals=proposals,
                text_features=text_features,
                relation_ids=relation_ids,
                proposal_masks=proposal_masks
            )
            
            # Loss
            _, loss_dict = criterion(
                model_outputs=outputs,
                gt_boxes=gt_boxes,
                relation_ids=relation_ids,
                hard_negatives=hard_negatives,
                hard_negatives_mask=hard_negatives_mask,
                proposals=proposals
            )
            
            for k, v in loss_dict.items():
                total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v
                
            # Collect for validation metrics
            all_pred_boxes.append(outputs['refined_boxes'].cpu())
            all_gt_boxes.append(gt_boxes.cpu())
            all_pred_scores.append(outputs['scores'].cpu())
            
    # Concatenate results
    all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
    all_gt_boxes = torch.cat(all_gt_boxes, dim=0)
    all_pred_scores = torch.cat(all_pred_scores, dim=0)
    
    # Calculate grounding metrics
    eval_metrics = evaluate_grounding(all_pred_boxes, all_gt_boxes, all_pred_scores)
    
    # Add losses
    num_batches = len(dataloader)
    for k, v in total_loss_dict.items():
        eval_metrics[f'eval_{k}'] = v / num_batches
        
    # Restore original weights if EMA was applied
    if ema is not None:
        ema.restore()
        
    return eval_metrics
