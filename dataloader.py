import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class CustomDataLoader:
    def __init__(self, data_dir, train_ratio, test_ratio, val_ratio, seed=42):
        # Definir transformações de dados
        self.data_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Carregar conjunto de dados
        self.dataset = datasets.ImageFolder(data_dir, self.data_transforms)

        # Aplicar a semente
        torch.manual_seed(seed)

        # Calcular tamanhos de split
        train_len = int(train_ratio * len(self.dataset))
        test_len = int(test_ratio * len(self.dataset))
        val_len = len(self.dataset) - train_len - test_len

        # Dividir conjunto de dados
        self.train_data, self.test_data, self.val_data = random_split(
            self.dataset, [train_len, test_len, val_len]
        )

    def get_train_data(self):
        return DataLoader(self.train_data, batch_size=4, shuffle=True, num_workers=4)

    def get_test_data(self):
        return DataLoader(self.test_data, batch_size=4, shuffle=True, num_workers=4)

    def get_val_data(self):
        return DataLoader(self.val_data, batch_size=4, shuffle=True, num_workers=4)
