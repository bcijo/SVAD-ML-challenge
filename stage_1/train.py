import os
import argparse
import torch
from torch import nn, optim
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
sys.path.append("/Users/sidhaarthmurali/Desktop/SVAD-ML-challenge")
from dataset import get_dataloaders
from model import svadVLM
from util import ContrastiveLoss, count_parameters  # Assuming utils.py is properly set up
from safetensors.torch import save_file

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the hybrid model with contrastive loss.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing the dataset.')
    parser.add_argument('--vision_model_name', type=str, default='google/vit-base-patch16-224', help='Name or path of the vision model.')
    parser.add_argument('--language_model_name', type=str, default='openai-community/gpt2-xl', help='Name or path of the language model.')
    parser.add_argument('--num_learnable_queries', type=int, default=96, help='Number of learnable queries.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')  
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for contrastive loss.')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use.')
    parser.add_argument('--max_val_samples', type=int, default=None, help='Maximum number of validation samples to use.')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    return args

def initialize_tokenizer(language_model_name):
    tokenizer = AutoTokenizer.from_pretrained(language_model_name)
    special_tokens = {'pad_token': '[PAD]', 'sep_token': '<sep>'}
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = '[PAD]'
    tokenizer.sep_token = '<sep>'
    return tokenizer

def initialize_model(args, tokenizer, device):
    model = svadVLM(
        vision_model_name=args.vision_model_name,
        language_model_name=args.language_model_name,
        num_learnable_queries=args.num_learnable_queries,
        cross_attention_positions=[i for i in range(0, 12, 4)]  # Example positions
    )
    model.to(device)

    # Resize token embeddings to accommodate new tokens
    model.language_model.language_model.resize_token_embeddings(len(tokenizer))
    return model

def freeze_parameters(model, cross_attention_positions):
    # Freeze the vision encoder
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    # Freeze the base language model (excluding the cross-attention layers)
    for param in model.language_model.language_model.parameters():
        param.requires_grad = False

    # Unfreeze cross-attention layers and their layer norms
    for idx, block in enumerate(model.language_model.language_model.base_model.h):
        if idx in cross_attention_positions:
            if hasattr(block, 'crossattention') and block.crossattention is not None:
                for param in block.crossattention.parameters():
                    param.requires_grad = True
                for param in block.ln_cross_attn.parameters():
                    param.requires_grad = True

    # Ensure the learnable queries are trainable
    for param in model.learnable_queries.parameters():
        param.requires_grad = True

def get_trainable_parameters(model, cross_attention_positions):
    # Collect trainable parameters
    trainable_parameters = []
    # Collect cross-attention layers' parameters
    for idx, block in enumerate(model.language_model.language_model.base_model.h):
        if idx in cross_attention_positions:
            if hasattr(block, 'crossattention') and block.crossattention is not None:
                trainable_parameters.extend([param for param in block.crossattention.parameters() if param.requires_grad])
                trainable_parameters.extend([param for param in block.ln_cross_attn.parameters() if param.requires_grad])
    # Collect learnable queries' parameters
    trainable_parameters.extend([param for param in model.learnable_queries.parameters() if param.requires_grad])
    return trainable_parameters

def train_epoch(model, dataloader, output_dir, optimizer, contrastive_loss_fn, device):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc="Training", leave=False)
    for batch in loop:
        images = batch['image'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeddings = outputs[:, 0, :]  # Using the first token

        # Compute visual embeddings (if necessary)
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
    return average_loss

def main():
    # Parse arguments
    args = parse_arguments()
    output_dir = args.output_dir
    # Load model directly
    # from transformers import AutoTokenizer, AutoModelForCausalLM

    # tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")
    # model = AutoModelForCausalLM.from_pretrained("ai-forever/mGPT")

    # Device selection with MPS backend
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")

    tokenizer = initialize_tokenizer(args.language_model_name)
    model = initialize_model(args, tokenizer, device)
    cross_attention_positions = [i for i in range(0, 12, 4)]  # Example positions
    freeze_parameters(model, cross_attention_positions)
    count_parameters(model)

    # Get trainable parameters and optimizer
    trainable_parameters = get_trainable_parameters(model, cross_attention_positions)
    optimizer = optim.Adam(trainable_parameters, lr=args.learning_rate)
    contrastive_loss_fn = ContrastiveLoss(temperature=args.temperature)

    # Get DataLoaders with max samples
    train_dataloader, _ = get_dataloaders(
        csv_file=args.csv_file,
        batch_size=args.batch_size,
        tokenizer_model_id=args.language_model_name,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    # Training loop
    for epoch in range(args.num_epochs):
        # Training
        train_loss = train_epoch(model, train_dataloader, output_dir, optimizer, contrastive_loss_fn, device)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Average Training Loss: {train_loss:.4f}")

        # Save the model checkpoint after each epoch
        if output_dir:
            checkpoint_path = os.path.join(output_dir, f'svadVLM_epoch_{epoch + 1}.safetensors')
            try:
                save_file(model.state_dict(), checkpoint_path)
            except Exception as e:
                print(f"Error saving model checkpoint: {e}")

if __name__ == '__main__':
    main()
