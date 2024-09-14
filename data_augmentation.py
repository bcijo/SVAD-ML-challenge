
import os
import torch
import pandas as pd
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random

csv_path = 'train.csv'
df = pd.read_csv(csv_path)

entity_name_to_idx = {name: idx for idx, name in enumerate(df['entity_name'].unique())}
df['entity_name_idx'] = df['entity_name'].map(entity_name_to_idx)

test_counts =  {
    'height': 32282,
    'depth': 28146,
    'width': 26931,
    'item_weight': 22032,
    'maximum_weight_recommendation': 7028,
    'voltage': 5488,
    'wattage': 5447,
    'item_volume': 3833
}

train_counts = df['entity_name'].value_counts().to_dict()

train_counts

factor = 2.1

augmented_image_dir = 'augmented_images'
os.makedirs(augmented_image_dir, exist_ok=True)

sampled_data = []

to_pil = transforms.ToPILImage()

class ImageWithTransformationsDataset(Dataset):
    def __init__(self, metadata, img_size=1024, augment=False):
        self.metadata = metadata
        self.img_size = img_size
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5], [0.5])
        ])

        self.augmentation = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.Lambda(self.change_background_color),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5], [0.5])
        ])

    def change_background_color(self, img):
        if isinstance(img, torch.Tensor):
            raise TypeError("Input should be a PIL Image for color transformation")

        background_color = tuple([random.randint(0, 255) for _ in range(3)])
        img_with_bg = ImageOps.colorize(ImageOps.grayscale(img), black=background_color, white="white")
        return img_with_bg

    def download_image(self, url):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return img
        except Exception as e:
            print(f"Error downloading image {url}: {e}")
            return None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_url = self.metadata.iloc[idx]['image_link']
        entity_name_idx = int(self.metadata.iloc[idx]['entity_name_idx'])
        img = self.download_image(image_url)
        if img is None:
            return None

        img_transformed = self.transform(img)

        if self.augment:
            img_augmented = self.augmentation(img)
        else:
            img_augmented = img_transformed

        return img_transformed, img_augmented, torch.tensor(entity_name_idx)

for class_name, test_count in test_counts.items():
    train_count = train_counts.get(class_name, 0)
    target_count = int(factor * test_count)

    df_class = df[df['entity_name'] == class_name]

    if train_count > target_count:
        df_class_sampled = resample(df_class, replace=False, n_samples=target_count, random_state=42)
        sampled_data.append(df_class_sampled)

    else:
        sampled_data.append(df_class)
        num_augmentations = target_count - train_count

        augment_dataset = ImageWithTransformationsDataset(metadata=df_class, img_size=512, augment=True)

        for i in range(num_augmentations):
            random_idx = np.random.choice(len(augment_dataset))

            _, img_augmented, entity_name_idx = augment_dataset[random_idx]

            img_augmented_pil = to_pil(img_augmented)

            augmented_filename = f"{class_name}_augmented_{i}.png"
            augmented_image_path = os.path.join(augmented_image_dir, augmented_filename)

            img_augmented_pil.save(augmented_image_path)

            sampled_data.append({
                'image_link': augmented_image_path,
                'entity_name': class_name,
                'entity_name_idx': entity_name_idx.item()
            })