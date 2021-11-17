import torch
from torch.utils.data import Dataset
import torchvision.io as io
import torchvision.transforms.functional as AF
from pathlib import Path
import json


class iNaturalistDataset(Dataset):
    def __init__(self, metadata_filename, image_size, category_level="supercategory"):
        self.metadata_path = Path(metadata_filename).resolve()
        self.image_size = image_size
        self.aspect_ratio = self.image_size[0] / self.image_size[1]
        self.category_level = category_level
        with open(self.metadata_path, "r") as mf:
            metadata = json.load(mf)
        self.images = metadata["images"]
        self.categories = {cat["id"]: cat for cat in metadata["categories"]}
        labels = set((cat[self.category_level] for cat in self.categories.values()))
        self.labels_to_ord = {level: i for i, level in enumerate(labels)}
        self.ord_to_labels = {i: level for level, i in self.labels_to_ord.items()}
        if "annotations" in metadata:
            self.annotations = {ann["image_id"]: ann for ann in metadata["annotations"]}
        else:
            self.annotations = None

    def __getitem__(self, item):
        img_path = self.metadata_path.parent / Path(self.images[item]["file_name"])
        img = io.read_image(str(img_path.absolute()))
        img = AF.convert_image_dtype(img, torch.float)
        # take a center crop of the correct aspect ratio and resize
        crop_size = [img.shape[1], int(round(img.shape[1] * self.aspect_ratio))]
        img = AF.center_crop(img, crop_size)
        img = AF.resize(img, self.image_size)
        if self.annotations is not None:
            label = self.categories[self.annotations[self.images[item]["id"]]["category_id"]][self.category_level]
            return img, self.labels_to_ord[label]
        else:
            return img, -1

    def get_label_string(self, label):
        return self.ord_to_labels[label]

    def __len__(self):
        return len(self.images)
