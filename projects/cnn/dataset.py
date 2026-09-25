"""Load only eligible samples from the existing frozen audit manifest."""
import csv
from pathlib import Path


from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_TO_IDX = {'NORMAL': 0, 'PNEUMONIA': 1}


def make_transform(training=False, improved=False):
    operations = [transforms.Resize((100, 100))]
    if training and improved:
        operations.append(transforms.RandomAffine(degrees=7, translate=(0.05, 0.05)))
    operations.append(transforms.ToTensor())  # uint8 [0,255] -> float32 [0,1]
    return transforms.Compose(operations)


class ChestXrayDataset(Dataset):
    def __init__(self, data_root, manifest, split, improved=False):
        self.root = Path(data_root).resolve()
        with Path(manifest).open(newline='', encoding='utf-8') as file:
            self.samples = [row for row in csv.DictReader(file)
                            if row['status'] == 'eligible' and row['split'] == split]
        if not self.samples:
            raise ValueError(f'No eligible samples for {split}')
        self.labels = [int(row['label']) for row in self.samples]
        for row in self.samples:
            if CLASS_TO_IDX[row['class_name']] != int(row['label']):
                raise ValueError('Manifest label mapping does not match NORMAL=0/PNEUMONIA=1')
            path = (self.root / row['path']).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f'Missing or invalid image path: {row["path"]}')
        self.transform = make_transform(training=(split == 'train'), improved=improved)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        with Image.open(self.root / row['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, int(row['label'])
