import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

#TODO: Implement the CustomSkinCancerDataset class then look into adding this transform to it
class VignetteTransform:
    def __init__(self, intensity=0.5):
        self.intensity = intensity  # Controls the strength of the vignette effect

    def __call__(self, img):
        width, height = img.size
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xv, yv = np.meshgrid(x, y)
        radius = np.sqrt(xv**2 + yv**2)
        
        # Vignette mask
        mask = 1 - (radius ** 2) * self.intensity
        mask = np.clip(mask, 0, 1)
        
        # Convert mask to a PIL image
        mask = torch.tensor(mask).unsqueeze(2).repeat(1, 1, 3)


class CustomSkinCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None, training_preprocess=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.training_preprocess = training_preprocess

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Apply training_preprocess if any
        if self.training_preprocess:
            img = self.training_preprocess(img)
        
        # Apply FinalTransform 
        if self.transform:
            img = self.transform(img)
        
        return img, label

def pick_dataloader(path_to_your_dataset, batch_size=16):
    data_dir = path_to_your_dataset
    
    # Define preprocessing and transformations
    training_preprocess = transforms.Compose([
        # VignetteTransform(intensity=0.5),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ])

    transform = transforms.Compose([
        # transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    custom_dataset_train = CustomSkinCancerDataset(root_dir=data_dir, transform=transform, training_preprocess=training_preprocess)
    custom_dataset_test  = CustomSkinCancerDataset(root_dir=data_dir, transform=transform)

    # Split dataset into training and testing sets
    train_size = int(0.8 * len(custom_dataset_train))
    test_size = len(custom_dataset_train) - train_size
    train_dataset, _ = random_split(custom_dataset_train, [train_size, test_size])
    _, test_dataset = random_split(custom_dataset_test, [train_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader