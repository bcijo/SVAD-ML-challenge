import torch
from torch import nn
from stage_1.util import VisionEncoder, LanguageModelWithCrossAttention, LearnableQueries

class svadVLM(nn.Module):
    def __init__(
        self,
        vision_model_name='google/vit-base-patch16-224',
        language_model_name='gpt2',
        num_learnable_queries=24,
        cross_attention_positions=None
    ):
        super(svadVLM, self).__init__()

        self.vision_encoder = VisionEncoder(
            model_name=vision_model_name,
            output_dim=None  
        )
        self.learnable_queries = LearnableQueries(
            num_queries=num_learnable_queries,
            hidden_size=self.vision_encoder.output_dim
        )
        self.language_model = LanguageModelWithCrossAttention(
            model_name=language_model_name,
            cross_attention_positions=cross_attention_positions
        )

    def forward(self, images, input_ids, attention_mask=None):
        batch_size = images.size(0)

        visual_features = self.vision_encoder(images)  # (batch_size, num_patches + 1, hidden_size)
        queries = self.learnable_queries(batch_size)  # (batch_size, num_queries, hidden_size)


        encoder_hidden_states = torch.cat([queries, visual_features], dim=1)  # (batch_size, total_tokens, hidden_size)

        # Create encoder attention mask
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.size()[:-1],
            dtype=torch.long,
            device=encoder_hidden_states.device
        )

        # Pass through language model with cross-attention
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )

        return outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
