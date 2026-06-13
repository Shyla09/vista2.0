import torch
import torch.nn as nn
import math

try:
    from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class CLIPExtractor(nn.Module):
    """
    CLIP feature extractor (frozen) for VISTA 2.0.
    Dynamically supports ViT-B/32 and ViT-L/14 backbones.
    Auto-detects hidden dims and spatial grid shapes.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", project_dim=None):
        super().__init__()
        self.model_name = model_name
        self.vision_model = None
        self.text_model = None
        self.processor = None
        self.tokenizer = None
        
        # Default hidden dims
        self.vision_hidden_dim = 768
        self.text_hidden_dim = 512
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.vision_model = CLIPVisionModel.from_pretrained(model_name)
                self.text_model = CLIPTextModel.from_pretrained(model_name)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
                
                # Freeze parameters
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                for param in self.text_model.parameters():
                    param.requires_grad = False
                
                self.vision_model.eval()
                self.text_model.eval()
                
                self.vision_hidden_dim = self.vision_model.config.hidden_size
                self.text_hidden_dim = self.text_model.config.hidden_size
            except Exception as e:
                print(f"Warning: Failed to load CLIP model '{model_name}': {e}. Using fallback.")
                self.vision_model = None
                self.text_model = None
        
        # If project_dim is not specified, default to the vision hidden dim (768 for Base, 1024 for Large)
        self.project_dim = project_dim if project_dim is not None else self.vision_hidden_dim
        
        # Trainable projection for text encoder to match multimodal project_dim
        self.text_projection = nn.Linear(self.text_hidden_dim, self.project_dim)
        print(f"CLIP Extractor initialized: model={model_name}, vision_dim={self.vision_hidden_dim}, project_dim={self.project_dim}")

    def extract_visual_features(self, images, device=None):
        """
        Extract visual feature map from intermediate Layer 9 of CLIP Vision.
        Dynamically handles grid sizes (7x7 for base, 16x16 for large).
        Returns:
            features: Tensor of shape (B, project_dim, H_grid, W_grid)
        """
        if self.vision_model is None:
            batch_size = len(images) if isinstance(images, list) else images.shape[0]
            dev = device if device is not None else torch.device("cpu")
            # Default fallback grid size 7x7
            return torch.zeros((batch_size, self.vision_hidden_dim, 7, 7), device=dev)
        
        if isinstance(images, list):
            inputs = self.processor(images=images, return_tensors="pt").to(self.vision_model.device)
            pixel_values = inputs["pixel_values"]
        else:
            pixel_values = images.to(self.vision_model.device)
            
        with torch.no_grad():
            outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            # Layer 9 intermediate output: (B, Seq_Len, hidden_size)
            hidden_states = outputs.hidden_states[9]
            
            # Remove CLS token (index 0)
            patch_tokens = hidden_states[:, 1:, :]
            B, P, C = patch_tokens.shape
            
            # Compute grid size dynamically (e.g. sqrt(49) = 7, sqrt(256) = 16)
            grid_size = int(math.sqrt(P))
            assert grid_size * grid_size == P, f"Patches count {P} must be a perfect square"
            
            # Reshape patches to spatial feature map: (B, C, H_grid, W_grid)
            spatial_features = patch_tokens.transpose(1, 2).reshape(B, C, grid_size, grid_size)
            
        return spatial_features

    def extract_text_features(self, text_queries, device=None):
        """
        Extract text features from CLIP Text Encoder and project them to project_dim.
        Returns:
            dict with:
                - 'pooled': (B, project_dim)
                - 'sequence': (B, S, project_dim)
                - 'mask': (B, S)
        """
        if self.text_model is None:
            batch_size = len(text_queries)
            dev = device if device is not None else torch.device("cpu")
            dummy_pooled = torch.zeros((batch_size, self.text_hidden_dim), device=dev)
            dummy_seq = torch.zeros((batch_size, 10, self.text_hidden_dim), device=dev)
            
            projected_pooled = self.text_projection(dummy_pooled)
            projected_seq = self.text_projection(dummy_seq)
            mask = torch.ones((batch_size, 10), dtype=torch.float32, device=dev)
            
            return {
                "pooled": projected_pooled,
                "sequence": projected_seq,
                "mask": mask
            }
            
        inputs = self.tokenizer(text_queries, padding=True, truncation=True, return_tensors="pt").to(self.text_model.device)
        attention_mask = inputs["attention_mask"].float()
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            pooled_output = outputs.pooler_output
            sequence_output = outputs.last_hidden_state
            
        projected_pooled = self.text_projection(pooled_output)
        projected_seq = self.text_projection(sequence_output)
        
        return {
            "pooled": projected_pooled,
            "sequence": projected_seq,
            "mask": attention_mask
        }
