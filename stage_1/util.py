from torch import nn
from transformers import ViTModel, GPT2Model
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers import GPT2Model, GPT2Config

class VisionEncoder(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224', output_dim=None):
        super(VisionEncoder, self).__init__()
        self.vision_model = ViTModel.from_pretrained(model_name)
        self.visual_hidden_size = self.vision_model.config.hidden_size

        # Projection layer if output_dim is specified
        if output_dim and self.visual_hidden_size != output_dim:
            self.visual_projection = nn.Linear(self.visual_hidden_size, output_dim)
            self.output_dim = output_dim
        else:
            self.visual_projection = nn.Identity()
            self.output_dim = self.visual_hidden_size

    def forward(self, images):
        outputs = self.vision_model(pixel_values=images)
        visual_features = outputs.last_hidden_state  # (batch_size, num_patches + 1, visual_hidden_size)
        visual_features = self.visual_projection(visual_features)
        return visual_features  # (batch_size, num_patches + 1, output_dim)
    

class LanguageModelWithCrossAttention(nn.Module):
    def __init__(self, model_name='gpt2', cross_attention_positions=None):
        super(LanguageModelWithCrossAttention, self).__init__()
        # Load the GPT-2 configuration and set add_cross_attention=True
        config = GPT2Config.from_pretrained(model_name)
        config.add_cross_attention = True  # Enable cross-attention globally
        self.language_model = GPT2Model.from_pretrained(model_name, config=config)
        self.hidden_size = self.language_model.config.hidden_size

        # Add cross-attention layers
        self._add_cross_attention_layers(cross_attention_positions)

    def _add_cross_attention_layers(self, cross_attention_positions):
        if cross_attention_positions is None:
            cross_attention_positions = [i for i in range(0, len(self.language_model.h), 4)]
        for idx, block in enumerate(self.language_model.h):
            if idx in cross_attention_positions:
                block.add_cross_attention = True
                block.crossattention = GPT2Attention(
                    self.language_model.config, is_cross_attention=True
                )
                block.ln_cross_attn = nn.LayerNorm(
                    self.hidden_size, eps=self.language_model.config.layer_norm_epsilon
                )
            else:
                block.add_cross_attention = False

    def forward(self, input_ids, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
        return outputs


class LearnableQueries(nn.Module):
    def __init__(self, num_queries, hidden_size):
        super(LearnableQueries, self).__init__()
        self.learnable_queries = nn.Parameter(torch.randn(num_queries, hidden_size))

    def forward(self, batch_size):
        # Expand queries to match batch size
        return self.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)


# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_fct = nn.CrossEntropyLoss()
        self.device = "cpu"
    def forward(self, query_embeddings, key_embeddings):
        batch_size = query_embeddings.size(0)
        query_embeddings = nn.functional.normalize(query_embeddings, p=2, dim=1)
        key_embeddings = nn.functional.normalize(key_embeddings, p=2, dim=1)
        logits = torch.matmul(query_embeddings, key_embeddings.T) / self.temperature
        labels = torch.arange(batch_size).to(self.device)
        loss = self.loss_fct(logits, labels)
        return loss
    
def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"Trainable parameters: {trainable_params}")
    print(f"Frozen parameters: {frozen_params}")
    
    return trainable_params, frozen_params

