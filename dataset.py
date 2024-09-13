import torch
from torch.utils.data import Dataset, DataLoader, random_split
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
            image = self.image_transform(image)
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

        sample = {
            'image': image,
            'entity_name_input_ids': entity_name_tokens['input_ids'].squeeze(0),
            'entity_name_attention_mask': entity_name_tokens['attention_mask'].squeeze(0),
            'entity_value_input_ids': entity_value_tokens['input_ids'].squeeze(0),
            'entity_value_attention_mask': entity_value_tokens['attention_mask'].squeeze(0)
        }
        return sample

def get_dataloaders(csv_file, batch_size=32, tokenizer_model_id='gpt2', val_split=0.2):
    dataset = CustomDataset(csv_file, tokenizer_model_id)
    
    # Calculate lengths for training and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders for train and validation sets
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