import torch
import pandas as pd
import requests
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from io import BytesIO
from transformers import GPT2Tokenizer

class TrainingDataset(Dataset):
    def __init__(self, data, tokenizer_model_id='gpt2'):
        self.data = data.reset_index(drop=True)
        
        # Image transformations for ViT
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ViT models typically use 224x224 images
            transforms.ToTensor(),
            # Normalize as per ViT requirements
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Initialize GPT-2 tokenizer with specified model ID
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model_id)
        # Add padding and separator tokens if not present
        special_tokens = {'pad_token': '[PAD]', 'sep_token': '<sep>'}
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token = '[PAD]'
        self.tokenizer.sep_token = '<sep>'
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.tokenizer.sep_token_id = self.tokenizer.convert_tokens_to_ids('<sep>')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the data row
        row = self.data.iloc[idx]

        # Load and process the image
        image = self.load_image(row['image_link'])

        # Get entity name and value
        entity_name = str(row['entity_name'])
        entity_value = str(row['entity_value'])
        
        # Combine entity_name and entity_value with a separator token
        combined_text = f"{entity_name} <sep> {entity_value}"

        # Tokenize the combined text
        tokens = self.tokenizer(
            combined_text,
            return_tensors='pt',
            padding='max_length',
            max_length=32,  # Adjust max_length as needed
            truncation=True
        )

        sample = {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        }
        return sample

    def load_image(self, image_url):
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image = self.image_transform(image)
        except Exception as e:
            print(f"Error loading image from {image_url}: {e}")
            # Return a tensor of zeros if image loading fails
            image = torch.zeros(3, 224, 224)
        return image

class ValidationDataset(Dataset):
    def __init__(self, data, tokenizer_model_id='gpt2'):
        self.data = data.reset_index(drop=True)
        
        # Image transformations for ViT
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ViT models typically use 224x224 images
            transforms.ToTensor(),
            # Normalize as per ViT requirements
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Initialize GPT-2 tokenizer with specified model ID
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model_id)
        # Add padding and separator tokens if not present
        special_tokens = {'pad_token': '[PAD]', 'sep_token': '<sep>'}
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token = '[PAD]'
        self.tokenizer.sep_token = '<sep>'
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.tokenizer.sep_token_id = self.tokenizer.convert_tokens_to_ids('<sep>')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the data row
        row = self.data.iloc[idx]

        # Load and process the image
        image = self.load_image(row['image_link'])

        # Get entity name and value
        entity_name = str(row['entity_name'])
        entity_value = str(row['entity_value'])

        # Tokenize entity_name
        entity_name_tokens = self.tokenizer(
            entity_name,
            return_tensors='pt',
            padding='max_length',
            max_length=16,  # Adjust max_length as needed
            truncation=True
        )
        # Tokenize entity_value
        entity_value_tokens = self.tokenizer(
            entity_value,
            return_tensors='pt',
            padding='max_length',
            max_length=32,
            truncation=True
        )

        sample = {
            'image': image,
            'entity_name_input_ids': entity_name_tokens['input_ids'].squeeze(0),
            'entity_name_attention_mask': entity_name_tokens['attention_mask'].squeeze(0),
            'entity_value_input_ids': entity_value_tokens['input_ids'].squeeze(0),
            'entity_value_attention_mask': entity_value_tokens['attention_mask'].squeeze(0)
        }

        return sample

    def load_image(self, image_url):
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image = self.image_transform(image)
        except Exception as e:
            print(f"Error loading image from {image_url}: {e}")
            # Return a tensor of zeros if image loading fails
            image = torch.zeros(3, 224, 224)
        return image

def get_dataloaders(csv_file, batch_size=32, tokenizer_model_id='gpt2', val_split=0.2, max_train_samples=None, max_val_samples=None):
    # Read the CSV file once
    full_data = pd.read_csv(csv_file)
    total_size = len(full_data)

    # Shuffle the data
    full_data = full_data.sample(frac=1).reset_index(drop=True)

    # Calculate lengths for training and validation
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Split the data into training and validation
    train_data = full_data.iloc[:train_size].reset_index(drop=True)
    val_data = full_data.iloc[train_size:].reset_index(drop=True)

    # Apply max_train_samples and max_val_samples
    if max_train_samples is not None:
        train_data = train_data.iloc[:max_train_samples].reset_index(drop=True)
    if max_val_samples is not None:
        val_data = val_data.iloc[:max_val_samples].reset_index(drop=True)

    # Create training and validation datasets
    train_dataset = TrainingDataset(train_data, tokenizer_model_id)
    val_dataset = ValidationDataset(val_data, tokenizer_model_id)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset and create DataLoader.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing the dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader.')
    parser.add_argument('--model_id', type=str, default='gpt2', help='Model ID for GPT-2 tokenizer.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of the dataset to use for validation.')

    args = parser.parse_args()

    # Create the DataLoaders for training and validation
    train_dataloader, val_dataloader = get_dataloaders(
        csv_file=args.csv_file,
        batch_size=args.batch_size,
        tokenizer_model_id=args.model_id,
        val_split=args.val_split
    )

    # For testing, iterate over one batch from the training dataloader
    print("Training data:")
    for batch in train_dataloader:
        images = batch['image']
        entity_name_input_ids = batch['entity_name_input_ids']
        entity_name_attention_mask = batch['entity_name_attention_mask']
        entity_value_input_ids = batch['entity_value_input_ids']
        entity_value_attention_mask = batch['entity_value_attention_mask']
        print(f"Batch images shape: {images.shape}")
        print(f"Entity name input IDs shape: {entity_name_input_ids.shape}")
        print(f"Entity value input IDs shape: {entity_value_input_ids.shape}")
        
    print("Validation data:")
    for batch in val_dataloader:
        images = batch['image']
        entity_name_input_ids = batch['entity_name_input_ids']
        entity_name_attention_mask = batch['entity_name_attention_mask']
        entity_value_input_ids = batch['entity_value_input_ids']
        entity_value_attention_mask = batch['entity_value_attention_mask']
        print(f"Batch images shape: {images.shape}")
        print(f"Entity name input IDs shape: {entity_name_input_ids.shape}")
        print(f"Entity value input IDs shape: {entity_value_input_ids.shape}")
        break  # Remove this break if you want to process the entire dataset