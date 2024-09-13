# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer
import argparse

class CustomDataset(Dataset):
    def __init__(self, csv_file, tokenizer_model_id='gpt2'):
        self.data = pd.read_csv(csv_file)
        
        # Image transformations for ViT
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ViT models typically use 224x224 images
            transforms.ToTensor(),
            # Normalize as per ViT requirements
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        breakpoint()
        
        # Initialize GPT-2 tokenizer with specified model ID
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model_id)
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = '[PAD]'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the data row
        row = self.data.iloc[idx]
        
        # Load and process the image
        image_url = row['image_link']
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            breakpoint()
            image = self.image_transform(image)
            breakpoint()
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a tensor of zeros if image loading fails
            image = torch.zeros(3, 224, 224)
        
        # Get entity name and value
        entity_name = str(row['entity_name'])
        entity_value = str(row['entity_value'])
        
        # Tokenize entity name and entity value
        entity_name_tokens = self.tokenizer(
            entity_name,
            return_tensors='pt',
            padding='max_length',
            max_length=32,
            truncation=True
        )
        entity_value_tokens = self.tokenizer(
            entity_value,
            return_tensors='pt',
            padding='max_length',
            max_length=32,
            truncation=True
        )
        breakpoint()
        sample = {
            'image': image,
            'entity_name_input_ids': entity_name_tokens['input_ids'].squeeze(0),
            'entity_name_attention_mask': entity_name_tokens['attention_mask'].squeeze(0),
            'entity_value_input_ids': entity_value_tokens['input_ids'].squeeze(0),
            'entity_value_attention_mask': entity_value_tokens['attention_mask'].squeeze(0)
        }
        return sample

def get_dataloader(csv_file, batch_size=32, tokenizer_model_id='gpt2'):
    dataset = CustomDataset(csv_file, tokenizer_model_id)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset and create DataLoader.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing the dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader.')
    parser.add_argument('--model_id', type=str, default='gpt2', help='Model ID for GPT-2 tokenizer.')

    args = parser.parse_args()

    # Create the DataLoader
    dataloader = get_dataloader(
        csv_file=args.csv_file,
        batch_size=args.batch_size,
        tokenizer_model_id=args.model_id
    )

    # For testing, iterate over one batch
    for batch in dataloader:
        images = batch['image']
        entity_name_input_ids = batch['entity_name_input_ids']
        entity_name_attention_mask = batch['entity_name_attention_mask']
        entity_value_input_ids = batch['entity_value_input_ids']
        entity_value_attention_mask = batch['entity_value_attention_mask']
        breakpoint()
        print(f"Batch images shape: {images.shape}")
        print(f"Entity name input IDs shape: {entity_name_input_ids.shape}")
        print(f"Entity value input IDs shape: {entity_value_input_ids.shape}")
        break  # Remove this break if you want to process the entire dataset
