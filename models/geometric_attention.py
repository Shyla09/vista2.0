import torch
import torch.nn as nn
import math

class SpatialBiasMLP(nn.Module):
    """
    VISTA 2.0 Spatial Bias Generator.
    Maps 11-dimensional pairwise bounding box geometries to attention head biases.
    Input size: 11 (dx, dy, |dx|, |dy|, dist, area_i, area_j, log_area_ratio, asp_i, asp_j, IoU)
    """
    def __init__(self, num_heads, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads)
        )

    def forward(self, boxes):
        """
        Args:
            boxes: Tensor of shape (B, N, 4) in normalized xyxy format.
        Returns:
            bias: Tensor of shape (B, num_heads, N, N)
        """
        B, N, _ = boxes.shape
        device = boxes.device
        
        # Extract coordinates
        xmin, ymin, xmax, ymax = boxes[:, :, 0], boxes[:, :, 1], boxes[:, :, 2], boxes[:, :, 3]
        
        # Width, Height, and Center coordinates
        w = torch.clamp(xmax - xmin, min=0.0)
        h = torch.clamp(ymax - ymin, min=0.0)
        
        cx = xmin + w / 2.0
        cy = ymin + h / 2.0
        area = w * h
        asp = torch.clamp(w / torch.clamp(h, min=1e-5), min=0.1, max=10.0)
        
        # Compute pairwise center offsets
        # cx.unsqueeze(2) is (B, N, 1) representing box i, cx.unsqueeze(1) is (B, 1, N) representing box j
        dx = cx.unsqueeze(2) - cx.unsqueeze(1)        # (B, N, N)
        dy = cy.unsqueeze(2) - cy.unsqueeze(1)        # (B, N, N)
        abs_dx = torch.abs(dx)
        abs_dy = torch.abs(dy)
        
        # 1. Euclidean Center Distance
        dist = torch.sqrt(dx**2 + dy**2 + 1e-6)       # (B, N, N)
        
        # 2. Areas and Scale Ratios
        area_i = area.unsqueeze(2).expand(-1, -1, N)  # (B, N, N)
        area_j = area.unsqueeze(1).expand(-1, N, -1)  # (B, N, N)
        log_area_ratio = torch.clamp(torch.log((area_i + 1e-5) / (area_j + 1e-5)), min=-5.0, max=5.0) # (B, N, N)
        
        # 3. Aspect Ratios
        asp_i = asp.unsqueeze(2).expand(-1, -1, N)    # (B, N, N)
        asp_j = asp.unsqueeze(1).expand(-1, N, -1)    # (B, N, N)
        
        # 4. Pairwise Intersection-over-Union (IoU) Overlap
        # Box coordinates: (B, N, 1, 2) and (B, 1, N, 2) for xmin/ymin, xmax/ymax comparisons
        lt = torch.max(boxes[:, :, :2].unsqueeze(2), boxes[:, :, :2].unsqueeze(1)) # (B, N, N, 2)
        rb = torch.min(boxes[:, :, 2:].unsqueeze(2), boxes[:, :, 2:].unsqueeze(1)) # (B, N, N, 2)
        wh = torch.clamp(rb - lt, min=0.0)            # (B, N, N, 2)
        inter = wh[:, :, :, 0] * wh[:, :, :, 1]       # (B, N, N)
        
        union = area_i + area_j - inter
        iou = inter / torch.clamp(union, min=1e-6)    # (B, N, N)
        
        # Stack all 11 geometric spatial relation features: (B, N, N, 11)
        spatial_features = torch.stack([
            dx, dy, abs_dx, abs_dy, dist,
            area_i, area_j, log_area_ratio,
            asp_i, asp_j, iou
        ], dim=-1)
        
        # Compute spatial bias G: (B, N, N, num_heads)
        bias = self.mlp(spatial_features)
        
        # Permute to (B, num_heads, N, N)
        bias = bias.permute(0, 3, 1, 2)
        
        return bias

class GeometricAttention(nn.Module):
    """
    Multi-Head Attention with Spatial Bias.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, spatial_bias, key_padding_mask=None):
        B, N, D = x.shape
        
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled logits: (B, H, N, N)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_logits = attn_logits + spatial_bias
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, N)
            # Safe negative value for FP16 mixed precision to avoid overflow
            attn_logits = attn_logits.masked_fill(mask == 0.0, -1e4)
            
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(context)
        
        return out

class GeometricAttentionLayer(nn.Module):
    """
    Transformer encoder layer with Spatial Bias.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.attn = GeometricAttention(embed_dim, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, spatial_bias, key_padding_mask=None):
        attn_out = self.attn(self.norm1(x), spatial_bias, key_padding_mask)
        x = x + self.dropout1(attn_out)
        
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout2(mlp_out)
        
        return x

class GeometricAttentionEncoder(nn.Module):
    """
    Encoder stack of N layers that generates spatial biases and refines representations.
    """
    def __init__(self, embed_dim=768, num_heads=8, num_layers=3, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.spatial_bias_generator = SpatialBiasMLP(num_heads)
        self.layers = nn.ModuleList([
            GeometricAttentionLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, boxes, masks=None):
        spatial_bias = self.spatial_bias_generator(boxes)
        
        for layer in self.layers:
            x = layer(x, spatial_bias, key_padding_mask=masks)
            
        return x

class RelationInjection(nn.Module):
    """
    Linguistic intention injector for spatial queries: x = x + rel_emb
    """
    def __init__(self, num_relations=6, embed_dim=768):
        super().__init__()
        self.rel_embedding = nn.Embedding(num_relations, embed_dim)
        
    def forward(self, x, relation_ids):
        rel_emb = self.rel_embedding(relation_ids) # (B, D)
        x_injected = x + rel_emb.unsqueeze(1)      # (B, N, D)
        return x_injected
