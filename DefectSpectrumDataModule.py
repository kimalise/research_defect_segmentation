import pytorch_lightning as pl
from DefectSpectrumDataset import DefectSpectrumDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DefectSpectrumDataModule(pl.LightningDataModule):
    def __init__(self, train_meta_json_path, val_meta_json_path, batch_size):
        super().__init__()
        self.train_meta_json_path = train_meta_json_path
        self.val_meta_json_path = val_meta_json_path
        self.batch_size = batch_size
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = DefectSpectrumDataset(self.train_meta_json_path, self.transform)

