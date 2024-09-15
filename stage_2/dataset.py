import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
from transformers import GPT2Tokenizer

class FineTuningDataset(Dataset):
    def __init__(self, data, tokenizer_model_id='gpt2', max_seq_length=16):
        self.data = data.reset_index(drop=True)
        # Image transformations
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalize as per ViT requirements
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_model_id)
        special_tokens = {'pad_token': '[PAD]', 'sep_token': '<sep>'}
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token = '[PAD]'
        self.tokenizer.sep_token = '<sep>'
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids('<sep>')
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        image = self.load_image(row['image_link'])
        
        # Get entity name and value
        entity_name = str(row['entity_name'])
        entity_value = str(row['entity_value'])
        
        # Tokenize entity name
        entity_name_tokens = self.tokenizer.encode(
            entity_name,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length - 1  # Reserve space for <sep>
        )
        
        # Build input_ids
        input_ids = entity_name_tokens + [self.sep_token_id]
        
        # Pad input_ids to max_seq_length
        input_ids = input_ids[:self.max_seq_length]
        input_ids += [self.pad_token_id] * (self.max_seq_length - len(input_ids))
        
        # Build labels
        labels = [-100] * self.max_seq_length
        
        # Tokenize entity value
        entity_value_tokens = self.tokenizer.encode(
            entity_value,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length - len(input_ids)
        )
        
        # Place entity_value_tokens into labels starting after the entity name and <sep>
        start_idx = len(entity_name_tokens) + 1  # +1 for <sep>
        end_idx = start_idx + len(entity_value_tokens)
        
        # Ensure end_idx does not exceed max_seq_length
        end_idx = min(end_idx, self.max_seq_length)
        labels[start_idx:end_idx] = entity_value_tokens[:end_idx - start_idx]
        
        # Pad labels with pad_token_id (optional, for clarity)
        labels = labels[:self.max_seq_length]
        labels += [self.pad_token_id] * (self.max_seq_length - len(labels))
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        sample = {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return sample
    
    def load_image(self, image_url):
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image = self.image_transform(image)
        except Exception as e:
            print(f"Error loading image from {image_url}: {e}")
            image = torch.zeros(3, 224, 224)
        return image


def get_dataloaders(csv_file, batch_size=32, tokenizer_model_id='gpt2', val_split=0.2, max_train_samples=None, max_val_samples=None):
    full_data = pd.read_csv(csv_file)
    total_size = len(full_data)

    full_data = full_data.sample(frac=1).reset_index(drop=True)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_data = full_data.iloc[:train_size].reset_index(drop=True)
    val_data = full_data.iloc[train_size:].reset_index(drop=True)

    if max_train_samples is not None:
        train_data = train_data.iloc[:max_train_samples].reset_index(drop=True)
    if max_val_samples is not None:
        val_data = val_data.iloc[:max_val_samples].reset_index(drop=True)

    train_dataset = FineTuningDataset(train_data, tokenizer_model_id)
    val_dataset = FineTuningDataset(val_data, tokenizer_model_id)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


    return train_dataloader, val_dataloader
