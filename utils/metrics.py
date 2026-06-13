import torch

def compute_iou(boxA, boxB):
    """
    Compute IoU between two tensors of boxes of shape (B, 4).
    """
    xA = torch.max(boxA[:, 0], boxB[:, 0])
    yA = torch.max(boxA[:, 1], boxB[:, 1])
    xB = torch.min(boxA[:, 2], boxB[:, 2])
    yB = torch.min(boxA[:, 3], boxB[:, 3])

    interArea = torch.clamp(xB - xA, min=0.0) * torch.clamp(yB - yA, min=0.0)
    
    boxAArea = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
    boxBArea = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])

    unionArea = boxAArea + boxBArea - interArea
    iou = interArea / torch.clamp(unionArea, min=1e-6)
    
    return iou

def evaluate_grounding(pred_boxes, gt_boxes, pred_scores):
    """
    Evaluates grounding performance.
    Args:
        pred_boxes: Refined bounding boxes of shape (B, N, 4) in normalized xyxy format
        gt_boxes: Ground truth boxes of shape (B, 4) in normalized xyxy format
        pred_scores: Grounding scores (logits) of shape (B, N)
    Returns:
        metrics: Dictionary containing:
            - 'acc_0.5': Accuracy @ IoU >= 0.5
            - 'acc_0.25': Accuracy @ IoU >= 0.25
            - 'mean_iou': Mean IoU
            - 'ious': List of IoU values
    """
    B, N, _ = pred_boxes.shape
    device = pred_boxes.device
    
    # 1. Identify the highest-scoring proposal index for each item in the batch
    best_indices = torch.argmax(pred_scores, dim=1) # (B,)
    
    # 2. Extract corresponding predicted boxes
    selected_boxes = pred_boxes[torch.arange(B, device=device), best_indices] # (B, 4)
    
    # 3. Compute IoU for each item
    ious = compute_iou(selected_boxes, gt_boxes) # (B,)
    
    # 4. Calculate metrics
    acc_05 = (ious >= 0.5).float().mean().item()
    acc_025 = (ious >= 0.25).float().mean().item()
    mean_iou = ious.mean().item()
    
    return {
        'acc_0.5': acc_05,
        'acc_0.25': acc_025,
        'mean_iou': mean_iou,
        'ious': ious.tolist()
    }
