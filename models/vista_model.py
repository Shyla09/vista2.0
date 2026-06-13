import torch
import torch.nn as nn

try:
    from torchvision.ops import roi_align
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from models.geometric_attention import GeometricAttentionEncoder, RelationInjection
from models.cross_attention import CrossAttentionLayer

class VISTAModel(nn.Module):
    """
    VISTA 2.0 Core Model (Visual Spatial Grounding with Geometric Attention)
    Supports dynamic projection channels (768 or 1024) and dynamic spatial grids.
    """
    def __init__(self, embed_dim=768, num_heads=8, num_layers=3, num_relations=6, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # BBox Coordinate Encoder: MLP 4 -> embed_dim
        self.bbox_encoder = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Geometric Attention Encoder: Stack of self-attention layers with 11-dim spatial MLP G
        self.geometric_encoder = GeometricAttentionEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Relation Token Injector
        self.relation_injection = RelationInjection(
            num_relations=num_relations,
            embed_dim=embed_dim
        )
        
        # Cross-Attention Layer: Objects attend to text tokens
        self.cross_attention = CrossAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output Heads
        self.object_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        self.box_regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 4)
        )
        
        self.relation_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_relations)
        )
        
        # Adaptive pooling to reduce crops to 1x1 before flatting
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, clip_visual_map, proposals, text_features, relation_ids, proposal_masks=None):
        """
        Args:
            clip_visual_map: Feature map tensor of shape (B, C_feat, H_feat, W_feat) (e.g. B, 1024, 16, 16)
            proposals: Candidate bounding boxes (B, N, 4) in normalized xyxy format
            text_features: Dictionary of text features with keys 'sequence', 'pooled', 'mask'
            relation_ids: Target relation class (B,)
            proposal_masks: Binary mask of valid proposals (B, N)
        """
        B, N, _ = proposals.shape
        _, C_feat, H_feat, W_feat = clip_visual_map.shape
        
        # 1. Extract visual features for each proposal using RoI-Align
        if TORCHVISION_AVAILABLE:
            # Scale proposals from [0, 1] to feature map scale [0, W_feat] & [0, H_feat]
            scaled_boxes = []
            box_scale = torch.tensor([W_feat, H_feat, W_feat, H_feat], 
                                     device=proposals.device, dtype=proposals.dtype)
            for i in range(B):
                scaled_boxes.append(proposals[i] * box_scale)
            
            # RoI Align output: (B * N, C_feat, 7, 7)
            # Fixed crop size of 7x7 avoids excessive upsampling/blur artifacts on larger feature maps
            roi_feat = roi_align(
                clip_visual_map, 
                scaled_boxes, 
                output_size=(7, 7), 
                spatial_scale=1.0, 
                aligned=True
            )
        else:
            # Fallback for offline testing
            roi_feat = torch.zeros((B * N, C_feat, 7, 7), 
                                   device=proposals.device, dtype=proposals.dtype)
            mean_visual = clip_visual_map.mean(dim=(2, 3)).repeat_interleave(N, dim=0)
            roi_feat = roi_feat + mean_visual.unsqueeze(-1).unsqueeze(-1)
            
        # Spatial average pool (H_feat, W_feat) -> (1, 1)
        pooled_feat = self.pool(roi_feat).flatten(1) # (B * N, C_feat)
        visual_features = pooled_feat.reshape(B, N, self.embed_dim) # (B, N, embed_dim)
        
        # 2. Encode bounding box coordinates: (B, N, 4) -> (B, N, embed_dim)
        bbox_features = self.bbox_encoder(proposals)
        
        # 3. Combine visual features and box coordinate representations
        x = visual_features + bbox_features # (B, N, embed_dim)
        
        # 4. Geometric Attention Encoder (Self-Attention with 11-dim spatial MLP G)
        x = self.geometric_encoder(x, proposals, masks=proposal_masks) # (B, N, embed_dim)
        
        # 5. Relation Token Injection
        x = self.relation_injection(x, relation_ids) # (B, N, embed_dim)
        
        # 6. Cross-Attention
        text_seq = text_features['sequence']
        text_mask = text_features.get('mask', None)
        x = self.cross_attention(x, text_seq, text_mask) # (B, N, embed_dim)
        
        # 7. Output heads
        scores = self.object_scorer(x).squeeze(-1)
        
        offsets = self.box_regressor(x)
        refined_boxes = self.refine_boxes(proposals, offsets)
        
        if proposal_masks is not None:
            weights = proposal_masks.unsqueeze(-1) # (B, N, 1)
            global_obj_feature = (x * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)
        else:
            global_obj_feature = x.mean(dim=1)
            
        relation_logits = self.relation_predictor(global_obj_feature)
        
        return {
            'scores': scores,
            'refined_boxes': refined_boxes,
            'relation_logits': relation_logits,
            'obj_features': x,
            'text_embed': text_features['pooled']
        }

    def refine_boxes(self, boxes, offsets):
        """
        Applies standard R-CNN/YOLO offsets (dx, dy, dw, dh) to proposal boxes.
        """
        xmin, ymin, xmax, ymax = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        
        px = (xmin + xmax) / 2.0
        py = (ymin + ymax) / 2.0
        pw = torch.clamp(xmax - xmin, min=1e-5)
        ph = torch.clamp(ymax - ymin, min=1e-5)
        
        dx, dy, dw, dh = offsets[..., 0], offsets[..., 1], offsets[..., 2], offsets[..., 3]
        
        rx = px + dx * pw
        ry = py + dy * ph
        rw = pw * torch.exp(torch.clamp(dw, min=-10.0, max=10.0))
        rh = ph * torch.exp(torch.clamp(dh, min=-10.0, max=10.0))
        
        rx_min = rx - rw / 2.0
        ry_min = ry - rh / 2.0
        rx_max = rx + rw / 2.0
        ry_max = ry + rh / 2.0
        
        refined = torch.stack([rx_min, ry_min, rx_max, ry_max], dim=-1)
        refined = torch.clamp(refined, min=0.0, max=1.0)
        
        return refined
