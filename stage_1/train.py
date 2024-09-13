import os
import argparse
import torch
from torch import nn, optim
from transformers import GPT2Tokenizer
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
    parser.add_argument('--language_model_name', type=str, default='gpt2', help='Name or path of the language model.')
    parser.add_argument('--num_learnable_queries', type=int, default=24, help='Number of learnable queries.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for contrastive loss.')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use.')
    parser.add_argument('--max_val_samples', type=int, default=None, help='Maximum number of validation samples to use.')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    return args

def initialize_tokenizer(language_model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(language_model_name)
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

def get_trainable_parameters(model, cross_attention_positions):
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
    return trainable_parameters

def train_epoch(model, dataloader, output_dir, optimizer, contrastive_loss_fn, device):
    model.train()
    total_loss = 0.0

    loop = tqdm(dataloader, desc="Training", leave=False)
    for batch in loop:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeddings = outputs[:, 0, :]  # Using the first token
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
        # IMP : Set the correct output directory here while training
    try:
        checkpoint_path = os.path.join(output_dir, 'svadVLM_final.safetensors')
        save_file(model.state_dict(), checkpoint_path)
    except Exception as e:
        print(f"Error saving model checkpoint: {e}")
    return average_loss

def validate_epoch(model, dataloader, tokenizer, device):
    model.eval()
    val_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            images = batch['image'].to(device)
            input_ids = batch['entity_name_input_ids'].to(device)
            attention_mask = batch['entity_name_attention_mask'].to(device)
            labels = batch['entity_value_input_ids'].to(device)

            # Prepare encoder_hidden_states from images
            visual_features = model.vision_encoder(images)
            batch_size = images.size(0)
            queries = model.learnable_queries(batch_size)
            encoder_hidden_states = torch.cat([queries, visual_features], dim=1)
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.size()[:-1],
                dtype=torch.long,
                device=device
            )

            # Shift labels for language modeling
            labels_shifted = labels.clone()
            labels_shifted[labels_shifted == tokenizer.pad_token_id] = -100  # Ignore padding tokens

            # Get model outputs
            outputs = model.language_model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels_shifted
            )

            # Compute loss
            loss = outputs.loss
            val_loss += loss.item()

            # Generate predictions
            generated_ids = model.language_model.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                max_length=labels.size(1) + input_ids.size(1),
                num_beams=1,
                early_stopping=True
            )

            # Extract predictions corresponding to the entity_value tokens
            predictions = generated_ids[:, input_ids.size(1):]  # Skip the input_ids
            total_samples += labels.size(0)

            # Compute accuracy
            for pred, label in zip(predictions, labels):
                # Compare predicted tokens with label tokens
                pred_text = tokenizer.decode(pred, skip_special_tokens=True).strip()
                label_text = tokenizer.decode(label, skip_special_tokens=True).strip()
                if pred_text == label_text:
                    correct_predictions += 1

    average_val_loss = val_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return average_val_loss, accuracy

def main():
    # Parse arguments
    args = parse_arguments()
    output_dir = args.output_dir
    # Global device variable
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    train_dataloader, val_dataloader = get_dataloaders(
        csv_file=args.csv_file,
        batch_size=args.batch_size,
        tokenizer_model_id=args.language_model_name,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )

    # Training and Validation loops
    for epoch in range(args.num_epochs):
        # Training
        train_loss = train_epoch(model, train_dataloader, output_dir, optimizer, contrastive_loss_fn, device)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Average Training Loss: {train_loss:.4f}")

        # Validation
        # val_loss, accuracy = validate_epoch(model, val_dataloader, tokenizer, device)
        # print(f"Epoch [{epoch + 1}/{args.num_epochs}], Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
