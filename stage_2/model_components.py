import torch
from torch import nn
from transformers import ViTModel, GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from peft import get_peft_model, LoraConfig, TaskType

class VisionEncoder(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224', output_dim=1600):
        super(VisionEncoder, self).__init__()
        self.vision_model = ViTModel.from_pretrained(model_name)
        self.visual_hidden_size = self.vision_model.config.hidden_size
        output_dim = 1600
        # Projection layer if output_dim is specified
        if output_dim and self.visual_hidden_size != output_dim:
            self.visual_projection = nn.Linear(self.visual_hidden_size, 1600)
            self.output_dim = 1600
        else:
            self.visual_projection = nn.Identity()
            self.output_dim = self.visual_hidden_size

    def forward(self, images):
        outputs = self.vision_model(pixel_values=images)
        visual_features = outputs.last_hidden_state  # (batch_size, num_patches + 1, visual_hidden_size)
        visual_features = self.visual_projection(visual_features)  # (batch_size, num_patches + 1, output_dim)
        return visual_features  # (batch_size, num_patches + 1, output_dim)

class LearnableQueries(nn.Module):
    def __init__(self, num_queries, hidden_size):
        super(LearnableQueries, self).__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_size))

    def forward(self, batch_size):
        return self.queries.expand(batch_size, -1, -1)

class LanguageModelWithLoRA(nn.Module):
    def __init__(self, model_name='openai-community/gpt2-xl', lora_config=None, cross_attention_positions=None):
        super(LanguageModelWithLoRA, self).__init__()
        config = GPT2Config.from_pretrained(model_name)
        config.add_cross_attention = True

        self.language_model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        self.hidden_size = self.language_model.config.hidden_size

        if lora_config is not None:
            self.language_model = get_peft_model(self.language_model, lora_config)
        
        self._configure_cross_attention_layers(cross_attention_positions)

    def _configure_cross_attention_layers(self, cross_attention_positions):
        if cross_attention_positions is None:
            cross_attention_positions = [i for i in range(0, len(self.language_model.transformer.h), 4)]

        for idx, block in enumerate(self.language_model.transformer.h):
            if idx in cross_attention_positions:
                block.add_cross_attention = True
                if not hasattr(block, 'crossattention') or block.crossattention is None:
                    block.crossattention = GPT2Attention(
                        self.language_model.config, is_cross_attention=True
                    )
                if not hasattr(block, 'ln_cross_attn') or block.ln_cross_attn is None:
                    block.ln_cross_attn = nn.LayerNorm(
                        self.hidden_size, eps=self.language_model.config.layer_norm_epsilon
                    )

    def forward(self, input_ids, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None):
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, **kwargs):
        return self.language_model.generate(**kwargs)

class svadVLM_LoRA(nn.Module):
    def __init__(self, vision_model_name, language_model_name, num_learnable_queries, cross_attention_positions):
        super(svadVLM_LoRA, self).__init__()

        # Initialize vision encoder (without LoRA)
        self.vision_encoder = VisionEncoder(vision_model_name)

        # Initialize learnable queries
        self.learnable_queries = LearnableQueries(num_learnable_queries, self.vision_encoder.output_dim)

        # LoRA configuration for the language model
        lora_config_lm = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=['c_attn', 'c_proj'],  # Apply LoRA to specific modules
            lora_dropout=0.1,
            bias='none',
            task_type=TaskType.CAUSAL_LM
        )

        # Initialize language model with LoRA
        self.language_model = LanguageModelWithLoRA(
            model_name=language_model_name,
            lora_config=lora_config_lm,
            cross_attention_positions=cross_attention_positions
        )

    def forward(self, images, input_ids, attention_mask, labels=None):
        # Process images
        visual_features = self.vision_encoder(images)
        batch_size = images.size(0)
        queries = self.learnable_queries(batch_size)
        encoder_hidden_states = torch.cat([queries, visual_features], dim=1)
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.size()[:-1],
            dtype=torch.long,
            device=encoder_hidden_states.device
        )

        # Language model forward
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, max_length, **generate_kwargs):
        generated_ids = self.language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            max_length=max_length,
            **generate_kwargs
        )
        return generated_ids


def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"Trainable parameters: {trainable_params}")
    print(f"Frozen parameters: {frozen_params}")

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



    