from torch.utils.data import Dataset, DataLoader
import cv2
import torch

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image)
            image = torch.tensor(image)

        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_paths)
    
class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        super(CustomDataLoader, self).__init__(dataset, batch_size = batch_size, shuffle=shuffle, num_workers=num_workers)