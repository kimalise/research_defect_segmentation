import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2
import json
import numpy as np

'''
    root_dir: 数据集根目录，例如:D:\worksapce\python\Defect_Spectrum\DS-VISION\Ring
'''

'''
    root_dir: 数据集的根目录
    output: meta_json
'''
def build_meta(root_dir):
    image_path = os.path.join(root_dir, 'image')
    mask_path = os.path.join(root_dir, 'mask')
    rbg_mask_path = os.path.join(root_dir, 'rbg_mask')

    mask_ext = os.path.splitext(mask_path[0])[1]
    rbg_mask_ext = os.path.splitext(rbg_mask_path[0])[1]

    image_filename_list = os.listdir(image_path)
    mask_filename_list = os.listdir(mask_path)
    rbg_filename_list = os.listdir(rbg_mask_path)

    result = []
    for image_file in image_filename_list:
        main_file_name = os.path.splitext(image_file)[0]

        mask_file_name = main_file_name + mask_ext
        rbg_mask_file_name = main_file_name + rbg_mask_ext

        if mask_file_name not in rbg_filename_list:
            print("cannot find {}".format(mask_file_name))
            continue

        if rbg_mask_ext not in rbg_filename_list:
            print("cannot find {}".format(rbg_mask_ext))
            continue

        result.append({
            "image": image_file,
            "mask": mask_file_name,
            "rbg_mask": rbg_mask_file_name
        })

    return result

def split_data(meta_json, train_rate):
    # with open(origin_meta_json_path, "r", encoding='utf-8') as f:
    #     meta_json = json.load(f)

    num_examples = len(meta_json)
    indices = np.arange(num_examples)
    np.random.shuffle(indices)

    meta_json = np.array(meta_json)
    meta_json = meta_json[indices]

    num_train = int(num_examples * train_rate)
    train_meta_json = meta_json[:num_train]
    val_meta_json = meta_json[num_train:]

    return train_meta_json, val_meta_json

class DefectSpectrumDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.mask_files = []
        self.rgb_mask_files = []
        for file in os.listdir(os.path.join(self.root_dir, 'image')):
            self.image_files.append(file)

        for file in os.listdir(os.path.join(self.root_dir, 'mask')):
            self.mask_files.append(file)

        for file in os.listdir(os.path.join(self.root_dir, 'rbg_mask')):
            self.rgb_mask_files.append(file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, "image", self.image_files[index])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask_path = os.path.join(self.root_dir, "mask", self.mask_files[index])
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']

        return aug_image, aug_mask

'''
    v2版本是在meta_json已经准备好的情况下的数据集
    meta_json可以分为训练集，验证集，测试集
'''
class DefectSpectrumDatasetV2(Dataset):
    def __init__(self, meta_json_path, transform=None):
        self.meta_json_path = meta_json_path
        with open(meta_json_path, "r") as f:
            self.meta_json = json.load(f)

        self.transform = transform
        self.example = [(example['image_filename'], example['mask_filename']) for example in self.meta_json]

    def __len__(self):
        return len(self.example)

    def __getitem__(self, index):
        image = cv2.imread(self.example[index][0], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(self.example[index][1], cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']

        return aug_image, aug_mask



