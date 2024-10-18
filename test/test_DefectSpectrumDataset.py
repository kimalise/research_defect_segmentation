from idlelib.iomenu import encoding
import os
from albumentations.pytorch import ToTensorV2
import json
from DefectSpectrumDataset import DefectSpectrumDataset, build_meta, split_data
from DefectSpectrumDataset import DefectSpectrumDatasetV2, save_json
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from common import *

transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# dataset = DefectSpectrumDataset(root_dir, transform=transform)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
#
# for step, batch in enumerate(dataloader):
#     print(batch)
#     break

meta_json = build_meta(root_dir)
train_rate = 0.8
train_meta_json, val_meta_json = split_data(meta_json, train_rate)

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

save_json(train_meta_json, os.path.join(run_dir, "train_meta.json"))
save_json(val_meta_json, os.path.join(run_dir, "val_meta.json"))

train_dataset = DefectSpectrumDatasetV2(os.path.join(run_dir, "train_meta.json"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

for step, batch in enumerate(train_loader):
    print(batch)
    break