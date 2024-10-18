from albumentations.pytorch import ToTensorV2

from DefectSpectrumDataset import DefectSpectrumDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

root_dir = "D:/worksapce/python/Defect_Spectrum/DS-VISION/Ring"

transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

dataset = DefectSpectrumDataset(root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

for step, batch in enumerate(dataloader):
    print(batch)
    break

