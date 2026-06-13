import torch
import torch.nn as nn
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class YoloDetector(nn.Module):
    """
    YOLOv8 proposal generator.
    Extracts object bounding boxes from images.
    """
    def __init__(self, model_name="yolov8n.pt", max_proposals=20, conf_threshold=0.25):
        super().__init__()
        self.max_proposals = max_proposals
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self.yolo = None
        
        if YOLO_AVAILABLE:
            try:
                self.yolo = YOLO(model_name)
                # Freeze YOLO parameters
                for param in self.yolo.model.parameters():
                    param.requires_grad = False
                self.yolo.model.eval()
            except Exception as e:
                print(f"Warning: Failed to load YOLOv8 model '{model_name}': {e}. Using fallback.")
                self.yolo = None
        else:
            print("Warning: 'ultralytics' package not available. YOLO detector running in fallback/dummy mode.")

    def forward(self, images, gt_boxes=None):
        """
        Extract proposals for a batch of images.
        Args:
            images: List of PIL Images or list of tensors of shape (B, C, H, W).
            gt_boxes: Tensor of ground truth bounding boxes of shape (B, 4) in normalized xyxy format.
        """
        batch_size = len(images) if isinstance(images, list) else images.shape[0]
        device = gt_boxes.device if gt_boxes is not None else torch.device("cpu")
        
        batch_proposals = []
        batch_classes = []
        batch_masks = []
        
        for i in range(batch_size):
            img = images[i]
            if isinstance(img, torch.Tensor):
                img_h, img_w = img.shape[1], img.shape[2]
            else:
                img_w, img_h = img.size
                
            boxes = []
            class_ids = []
            
            if self.yolo is not None:
                with torch.no_grad():
                    # Ensure input is BCHW (1, C, H, W)
                    yolo_input = img.unsqueeze(0) if isinstance(img, torch.Tensor) else img
                    results = self.yolo(yolo_input, verbose=False, conf=self.conf_threshold)[0]
                    
                    if results.boxes is not None and len(results.boxes) > 0:
                        det_boxes = results.boxes.xyxy.cpu().numpy()
                        det_classes = results.boxes.cls.cpu().numpy()
                        
                        for box, cls_id in zip(det_boxes, det_classes):
                            norm_box = [
                                box[0] / img_w,
                                box[1] / img_h,
                                box[2] / img_w,
                                box[3] / img_h
                            ]
                            norm_box = [max(0.0, min(1.0, coord)) for coord in norm_box]
                            boxes.append(norm_box)
                            class_ids.append(int(cls_id))
            
            if gt_boxes is not None:
                gt_box = gt_boxes[i].tolist()
                gt_already_proposed = False
                for box in boxes:
                    iou = self._compute_iou(gt_box, box)
                    if iou > 0.9:
                        gt_already_proposed = True
                        break
                
                if not gt_already_proposed:
                    boxes.insert(0, gt_box)
                    class_ids.insert(0, -1)
            
            num_proposals = len(boxes)
            if num_proposals > self.max_proposals:
                boxes = boxes[:self.max_proposals]
                class_ids = class_ids[:self.max_proposals]
                mask = [1.0] * self.max_proposals
            else:
                mask = [1.0] * num_proposals + [0.0] * (self.max_proposals - num_proposals)
                if num_proposals == 0:
                    boxes.append([0.0, 0.0, 1.0, 1.0])
                    class_ids.append(-1)
                    mask[0] = 1.0
                    num_proposals = 1
                
                while len(boxes) < self.max_proposals:
                    boxes.append([0.0, 0.0, 0.0, 0.0])
                    class_ids.append(-1)
            
            batch_proposals.append(boxes)
            batch_classes.append(class_ids)
            batch_masks.append(mask)
            
        proposals_t = torch.tensor(batch_proposals, dtype=torch.float32, device=device)
        classes_t = torch.tensor(batch_classes, dtype=torch.long, device=device)
        masks_t = torch.tensor(batch_masks, dtype=torch.float32, device=device)
        
        return proposals_t, classes_t, masks_t

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        unionArea = boxAArea + boxBArea - interArea
        if unionArea == 0.0:
            return 0.0
        return interArea / unionArea
