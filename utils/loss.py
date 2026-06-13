import torch
import torch.nn as nn
import torch.nn.functional as F

class VISTALoss(nn.Module):
    """
    Combines the grounding cross-entropy loss, auxiliary relation loss,
    InfoNCE contrastive loss (with hard negatives), and GIoU box regression loss.
    
    L_total = L_CE + lambda_rel * L_rel + lambda_nce * L_NCE + lambda_box * L_box
    """
    def __init__(self, lambda_rel=1.0, lambda_nce=1.0, lambda_box=5.0, temperature=0.07):
        super().__init__()
        self.lambda_rel = lambda_rel
        self.lambda_nce = lambda_nce
        self.lambda_box = lambda_box
        self.temperature = temperature
        
        # Label smoothed cross entropy for grounding classification
        self.ce_grounding = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.ce_relation = nn.CrossEntropyLoss()

    def forward(self, model_outputs, gt_boxes, relation_ids, hard_negatives, hard_negatives_mask, proposals):
        """
        Args:
            model_outputs: Dict output from VISTAModel
            gt_boxes: (B, 4) normalized ground truth boxes
            relation_ids: (B,) relation target IDs
            hard_negatives: (B, M, 4) normalized distractor boxes from dataset
            hard_negatives_mask: (B, M) binary mask indicating valid distractors
            proposals: (B, N, 4) proposal boxes passed to model
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary with individual loss components for logging
        """
        scores = model_outputs['scores']                 # (B, N)
        refined_boxes = model_outputs['refined_boxes']   # (B, N, 4)
        relation_logits = model_outputs['relation_logits'] # (B, num_relations)
        obj_features = model_outputs['obj_features']     # (B, N, embed_dim)
        
        device = scores.device
        B, N, _ = proposals.shape
        
        # 1. Determine target proposal index for each batch element (the one with highest IoU with GT)
        target_indices = []
        target_ious = []
        for i in range(B):
            ious = self._compute_batch_ious(proposals[i], gt_boxes[i].unsqueeze(0)).squeeze(1) # (N,)
            max_val, max_idx = torch.max(ious, dim=0)
            target_indices.append(max_idx)
            target_ious.append(max_val)
            
        target_indices = torch.stack(target_indices).to(device) # (B,)
        
        # 2. Main Grounding Loss (L_CE over N proposals)
        loss_ce = self.ce_grounding(scores, target_indices)
        
        # 3. Auxiliary Relation Loss (L_rel)
        loss_rel = self.ce_relation(relation_logits, relation_ids)
        
        # 4. Box Regression Loss (L_box: GIoU loss between target proposal's refined box and GT)
        # Extract refined target boxes: (B, 4)
        target_refined_boxes = refined_boxes[torch.arange(B, device=device), target_indices]
        loss_box = self._giou_loss(target_refined_boxes, gt_boxes)
        
        # 5. InfoNCE Loss (L_NCE) with Hard Negative distractor mining
        # Target feature: (B, embed_dim)
        target_obj_feats = obj_features[torch.arange(B, device=device), target_indices]
        
        # Text feature: (B, embed_dim)
        text_feats = model_outputs.get('text_embed', None)
        if text_feats is None:
            text_feats = target_obj_feats.clone() # fallback
            
        # L_NCE calculation
        # Normalize features
        target_obj_feats_norm = F.normalize(target_obj_feats, dim=-1)
        text_feats_norm = F.normalize(text_feats, dim=-1)
        
        # Base Similarity matrix: (B, B)
        sim_matrix = torch.matmul(target_obj_feats_norm, text_feats_norm.t()) / self.temperature
        
        loss_nce = 0.0
        for i in range(B):
            # Pos similarity for item i
            pos_sim = sim_matrix[i, i] # scalar
            
            # Negatives from other batch elements: (B-1,)
            neg_batch_sims = torch.cat([sim_matrix[i, :i], sim_matrix[i, i+1:]])
            
            # Negatives from same-image distractors (hard negatives)
            hard_neg_sims = []
            img_hard_neg_boxes = hard_negatives[i] # (M, 4)
            img_hard_neg_mask = hard_negatives_mask[i] # (M,)
            
            valid_hard_neg_count = int(img_hard_neg_mask.sum().item())
            if valid_hard_neg_count > 0:
                # Find matching proposal features
                for k in range(valid_hard_neg_count):
                    dist_box = img_hard_neg_boxes[k].unsqueeze(0) # (1, 4)
                    ious = self._compute_batch_ious(proposals[i], dist_box).squeeze(1) # (N,)
                    max_idx = torch.argmax(ious)
                    
                    # Get feature of this distractor proposal
                    dist_feat = obj_features[i, max_idx] # (embed_dim,)
                    dist_feat_norm = F.normalize(dist_feat, dim=-1)
                    
                    # Compute similarity with text query i
                    sim = torch.dot(dist_feat_norm, text_feats_norm[i]) / self.temperature
                    hard_neg_sims.append(sim)
                    
            if len(hard_neg_sims) > 0:
                hard_neg_sim_t = torch.stack(hard_neg_sims)
                all_negs = torch.cat([neg_batch_sims, hard_neg_sim_t])
            else:
                all_negs = neg_batch_sims
                
            # LogSumExp trick
            logits = torch.cat([pos_sim.unsqueeze(0), all_negs])
            loss_i = torch.logsumexp(logits, dim=0) - pos_sim
            loss_nce += loss_i
            
        loss_nce = loss_nce / B
        
        # 6. Combined Loss
        total_loss = loss_ce + self.lambda_rel * loss_rel + self.lambda_nce * loss_nce + self.lambda_box * loss_box
        
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_ce': loss_ce.item(),
            'loss_rel': loss_rel.item(),
            'loss_nce': loss_nce.item(),
            'loss_box': loss_box.item(),
            'mean_target_iou': torch.mean(torch.stack(target_ious)).item()
        }
        
        return total_loss, loss_dict

    def _compute_batch_ious(self, boxes1, boxes2):
        """
        Compute IoU between two sets of boxes of shapes (N, 4) and (M, 4).
        Returns matrix of shape (N, M).
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        area1 = torch.clamp(area1, min=0.0)
        area2 = torch.clamp(area2, min=0.0)
        
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
        
        wh = torch.clamp(rb - lt, min=0.0)  # (N, M, 2)
        inter = wh[:, :, 0] * wh[:, :, 1]   # (N, M)
        
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / torch.clamp(union, min=1e-6)
        
        return iou

    def _giou_loss(self, boxes1, boxes2):
        """
        Compute GIoU loss between boxes1 (B, 4) and boxes2 (B, 4).
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        area1 = torch.clamp(area1, min=0.0)
        area2 = torch.clamp(area2, min=0.0)
        
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = torch.clamp(rb - lt, min=0.0)
        inter = wh[:, 0] * wh[:, 1]
        
        union = area1 + area2 - inter
        iou = inter / torch.clamp(union, min=1e-6)
        
        lt_c = torch.min(boxes1[:, :2], boxes2[:, :2])
        rb_c = torch.max(boxes1[:, 2:], boxes2[:, 2:])
        wh_c = torch.clamp(rb_c - lt_c, min=0.0)
        area_c = wh_c[:, 0] * wh_c[:, 1]
        
        giou = iou - (area_c - union) / torch.clamp(area_c, min=1e-6)
        loss = 1.0 - giou
        
        return torch.mean(loss)
