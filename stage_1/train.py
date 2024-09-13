# train.py

import argparse
import torch
from torch import nn, optim
from transformers import GPT2Tokenizer
from tqdm import tqdm
import sys
sys.path.append("/Users/sidhaarthmurali/Desktop/SVAD-ML-challenge")
from dataset import get_dataloader
from model import svadVLM
from safetensors.torch import save_file
from stage_1.utils import ContrastiveLoss, count_parameters

def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description='Train the hybrid model with contrastive loss.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing the dataset.')
    parser.add_argument('--vision_model_name', type=str, default='google/vit-base-patch16-224', help='Name or path of the vision model.')
    parser.add_argument('--language_model_name', type=str, default='gpt2', help='Name or path of the language model.')
    parser.add_argument('--num_learnable_queries', type=int, default=24, help='Number of learnable queries.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for contrastive loss.')
    args = parser.parse_args()

    # Parameters
    csv_file = args.csv_file
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.language_model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'

    # Initialize the hybrid model
    model = svadVLM(
        vision_model_name=args.vision_model_name,
        language_model_name=args.language_model_name,
        num_learnable_queries=args.num_learnable_queries,
        cross_attention_positions=[i for i in range(0, 12, 4)]  # Example positions
    )
    model.to(device)

    # Resize token embeddings to accommodate new tokens
    model.language_model.language_model.resize_token_embeddings(len(tokenizer))

    # Freeze the vision encoder
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    # Freeze the base language model (excluding the cross-attention layers)
    for param in model.language_model.language_model.parameters():
        param.requires_grad = False

    # Unfreeze cross-attention layers and their layer norms
    cross_attention_positions = [i for i in range(0, len(model.language_model.language_model.h), 4)]
    for idx, block in enumerate(model.language_model.language_model.h):
        if idx in cross_attention_positions:
            if hasattr(block, 'crossattention') and block.crossattention is not None:
                for param in block.crossattention.parameters():
                    param.requires_grad = True
                for param in block.ln_cross_attn.parameters():
                    param.requires_grad = True

    # Ensure the learnable queries are trainable
    for param in model.learnable_queries.parameters():
        param.requires_grad = True

    # Count parameters
    count_parameters(model)

    # Collect trainable parameters
    trainable_parameters = []
    # Collect cross-attention layers' parameters
    for idx, block in enumerate(model.language_model.language_model.h):
        if idx in cross_attention_positions:
            if hasattr(block, 'crossattention') and block.crossattention is not None:
                trainable_parameters.extend([param for param in block.crossattention.parameters() if param.requires_grad])
                trainable_parameters.extend([param for param in block.ln_cross_attn.parameters() if param.requires_grad])
    # Collect learnable queries' parameters
    trainable_parameters.extend([param for param in model.learnable_queries.parameters() if param.requires_grad])

    optimizer = optim.Adam(trainable_parameters, lr=learning_rate)
    contrastive_loss_fn = ContrastiveLoss(temperature=args.temperature)

    # Get DataLoader
    dataloader = get_dataloader(
        csv_file=csv_file,
        batch_size=batch_size,
        tokenizer_model_id=args.language_model_name
    )

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Use tqdm for batches
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for batch in loop:
            images = batch['image'].to(device)
            input_ids = batch['entity_name_input_ids'].to(device)
            attention_mask = batch['entity_name_attention_mask'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_embeddings = outputs[:, 0, :]  # Using the first token
            breakpoint()
            with torch.no_grad():
                visual_features = model.vision_encoder(images)
                visual_embeddings = visual_features[:, 0, :]

            # Compute contrastive loss
            loss = contrastive_loss_fn(text_embeddings, visual_embeddings)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update tqdm description
            loop.set_postfix(loss=loss.item())

        average_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss:.4f}")
    
    # IMP : Set the correct output directory here while training
    output_dir = '/path/to/output/directory/'
    save_file(model.state_dict(), f'{output_dir}/svadVLM_final.safetensors')

if __name__ == '__main__':
    main()
