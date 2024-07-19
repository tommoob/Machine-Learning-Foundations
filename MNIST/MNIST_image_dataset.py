from torch.utils.data import Dataset
import torch


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        image = image.unsqueeze(0)
        label = self.labels[idx]
        return image, label