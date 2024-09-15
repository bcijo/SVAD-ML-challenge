import argparse
import torch
from torch import optim
from tqdm import tqdm
from transformers import GPT2Tokenizer
import sys
from safetensors.torch import load_file, save_file
from dataset import get_dataloaders
from model_components import svadVLM_LoRA, count_parameters

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine-tune the model using LoRA adapters.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing the dataset.')
    parser.add_argument('--model_dir', type=str, required=False, help='saved model path')
    parser.add_argument('--vision_model_name', type=str, default='google/vit-base-patch16-224', help='Name or path of the vision model.')
    parser.add_argument('--language_model_name', type=str, default='gpt2', help='Name or path of the language model.')
    parser.add_argument('--num_learnable_queries', type=int, default=24, help='Number of learnable queries.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio.')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples.')
    parser.add_argument('--max_val_samples', type=int, default=None, help='Maximum number of validation samples.')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
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

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.language_model_name)
    special_tokens = {'pad_token': '[PAD]', 'sep_token': '<sep>'}
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = '[PAD]'
    tokenizer.sep_token = '<sep>'

    # Initialize model
    model = svadVLM_LoRA(
        vision_model_name=args.vision_model_name,
        language_model_name=args.language_model_name,
        num_learnable_queries=args.num_learnable_queries,
        cross_attention_positions=[i for i in range(0, 12, 4)]
    )

    # Resize token embeddings
    model.language_model.language_model.resize_token_embeddings(len(tokenizer))

    # Load the aligned model weights if available
    # Uncomment and adjust the path if you have a saved model to load
    aligned_model_path = args.model_dir
    if aligned_model_path:
        state_dict = load_file(aligned_model_path)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    # Freeze base model parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze LoRA adapter parameters
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    # Count trainable parameters
    count_parameters(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # Prepare dataloaders
    train_dataloader, val_dataloader = get_dataloaders(
        csv_file=args.csv_file,
        batch_size=args.batch_size,
        tokenizer_model_id=args.language_model_name,
        val_split=args.val_split,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]", leave=False)
        for batch in loop:
            images = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Set labels to -100 where labels are pad_token_id to ignore loss on padding tokens
            labels[labels == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            loop.set_postfix(loss=loss.item())

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Average Training Loss: {average_loss:.4f}")

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            val_loop = tqdm(val_dataloader, desc="Validation", leave=False)
            for batch in val_loop:
                images = batch['image'].to(device, non_blocking=True)
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                # Prepare labels
                labels[labels == tokenizer.pad_token_id] = -100

                # Forward pass
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_val_loss += loss.item()

                # Generate predictions
                batch_size = images.size(0)
                visual_features = model.vision_encoder(images)
                queries = model.learnable_queries(batch_size)
                encoder_hidden_states = torch.cat([queries, visual_features], dim=1)
                encoder_attention_mask = torch.ones(
                    encoder_hidden_states.size()[:-1],
                    dtype=torch.long,
                    device=device
                )
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    max_length=input_ids.size(1) + 4,  # Limit to 4 generated tokens
                    num_beams=1,
                    early_stopping=True
                )
                # Extract generated tokens corresponding to the entity value
                generated_tokens = generated_ids[:, input_ids.size(1):]  # Skip the input_ids
                labels_tokens = batch['labels']

                # Compare generated tokens with labels
                for gen_toks, lbl_toks in zip(generated_tokens, labels_tokens):
                    # Remove padding and special tokens
                    gen_toks = gen_toks[gen_toks != tokenizer.pad_token_id]
                    lbl_toks = lbl_toks[lbl_toks != -100]

                    gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
                    lbl_text = tokenizer.decode(lbl_toks, skip_special_tokens=True).strip()
                    if gen_text == lbl_text:
                        total_correct += 1
                    total_samples += 1

            average_val_loss = total_val_loss / len(val_dataloader)
            val_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {average_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # Save the fine-tuned model
    save_file(model.state_dict(), 'saved_models/stage_2/model.safetensors')

if __name__ == '__main__':
    main()
          