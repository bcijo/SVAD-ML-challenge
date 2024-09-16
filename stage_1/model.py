import torch
from torch import nn
from stage_1.util import VisionEncoder, LanguageModelWithCrossAttention, LearnableQueries

class svadVLM(nn.Module):
    def __init__(
        self,
        vision_model_name='google/vit-base-patch16-224',
        language_model_name = '',
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

class NextTokenLoss(nn.Module):
    def __init__(self, vocab_size: int, loss_gen_type: str = "mixed", loss_gen_factor: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.loss_gen_factor = loss_gen_factor
        self.loss_gen_type = loss_gen_type
        
        if loss_gen_type == "token":
            # Token-level loss, sums the cross-entropy loss across the batch
            self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")
        elif loss_gen_type == "mixed":
            # Mixed loss, averages the cross-entropy loss across the batch
            self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        else:
            raise ValueError(f"Invalid loss_gen_type: {loss_gen_type}")

    def forward(self, logits, labels):
        # Shift logits and labels to compute next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Reshape logits and labels for cross-entropy loss
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)

        # Calculate loss based on the specified type (token or mixed)
        if self.loss_gen_type == "token":
            loss = self.cross_entropy(shift_logits, shift_labels) / labels.size(0)
        elif self.loss_gen_type == "mixed":
            loss = self.cross_entropy(shift_logits, shift_labels)

        return loss * self.loss_gen_factor