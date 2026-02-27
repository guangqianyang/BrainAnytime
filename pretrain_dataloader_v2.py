import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torchio as tio
from typing import List, Dict, Tuple, Optional
import random


class MultiModalPretrainDataset(Dataset):
    """
    多模态3D医学图像预训练数据集
    
    特点:
    - 支持多个数据集（A4, ADNIDOD, AIBL, BraTS, NACC）
    - 缺失模态填充为0，并提供observed_indicator
    - 支持数据增强（Spatial transforms）
    - 支持Modality Dropout增加组合多样性
    """
    
    # 统一的模态顺序
    MODALITY_ORDER = ['T1', 'T2', 'Flair', 'PET']  # 统一为4个模态
    
    # 每个数据集的模态列名映射到统一名称
    MODALITY_MAPPING = {
        'modality_data_A4.xlsx': {'T1': 'T1', 'T2': 'T2', 'Flair': 'Flair', 'Amy_PET': 'PET'},
        'modality_data_ADNIDOD.xlsx': {'T1': 'T1', 'T2': 'T2', 'Flair': 'Flair', 'PET': 'PET'},
        'modality_data_AIBL.xlsx': {'T1': 'T1', 'T2': 'T2', 'Flair': 'Flair', 'PET': 'PET'},
        'modality_data_BraTS.xlsx': {'T1w': 'T1', 'T2w': 'T2', 'Flair': 'Flair', 'PET': 'PET'},
        'modality_data_NACC.xlsx': {'T1': 'T1', 'T2': 'T2', 'Flair': 'Flair', 'Amyloid': 'PET'},
    }
    
    # Path prefix replacement: Excel paths use the old server prefix,
    # remap to the local data directory.
    OLD_PATH_PREFIX = "/home/data/Pretrain"
    NEW_PATH_PREFIX = "./data/Pretrain"

    def __init__(
        self,
        excel_dir: str = "./data/Match_data_path/pretraining_processed",
        image_size: Tuple[int, int, int] = (128, 128, 128),
        augmentation: bool = True,
        modality_dropout_prob: float = 0.3,
        min_modalities: int = 1,
        cache_data: bool = False,
    ):
        """
        Args:
            excel_dir: Excel文件目录路径
            image_size: 图像尺寸 (D, H, W)
            augmentation: 是否进行数据增强
            modality_dropout_prob: 每个模态被dropout的概率
            min_modalities: 至少保留的模态数量
            cache_data: 是否缓存加载的数据到内存
        """
        self.excel_dir = excel_dir
        self.image_size = image_size
        self.augmentation = augmentation
        self.modality_dropout_prob = modality_dropout_prob
        self.min_modalities = min_modalities
        self.cache_data = cache_data
        self.cache = {}
        
        # 加载所有样本
        self.samples = self._load_all_samples()
        print(f"Loaded {len(self.samples)} samples from {len(self.MODALITY_MAPPING)} datasets")
        
        # 初始化数据增强
        if self.augmentation:
            self.spatial_transform = tio.OneOf({
                tio.RandomFlip(axes=0, flip_probability=0.5): 0.33,
                tio.RandomAffine(scales=(0.9, 1.2), degrees=10, p=0.5): 0.33,
                tio.RandomElasticDeformation(
                    num_control_points=(10, 10, 10),
                    max_displacement=8,
                    locked_borders=2,
                    p=0.5
                ): 0.34,
            })
        
    def _load_all_samples(self) -> List[Dict]:
        """加载所有Excel文件中的样本"""
        samples = []
        
        for excel_file, modality_map in self.MODALITY_MAPPING.items():
            excel_path = os.path.join(self.excel_dir, excel_file)
            if not os.path.exists(excel_path):
                print(f"Warning: Excel file not found: {excel_path}")
                continue
                
            df = pd.read_excel(excel_path)
            dataset_name = excel_file.replace('modality_data_', '').replace('.xlsx', '')
            
            for idx, row in df.iterrows():
                sample = {
                    'dataset': dataset_name,
                    'subject_id': row.get('SubjectID', f'{dataset_name}_{idx}'),
                    'modalities': {}
                }
                
                # 映射模态路径
                for orig_col, unified_name in modality_map.items():
                    if orig_col in df.columns:
                        path = row[orig_col]
                        if pd.notna(path) and isinstance(path, str):
                            # Remap old server path prefix to local path
                            if path.startswith(self.OLD_PATH_PREFIX):
                                path = self.NEW_PATH_PREFIX + path[len(self.OLD_PATH_PREFIX):]
                            if os.path.exists(path):
                                sample['modalities'][unified_name] = path
                
                # 只添加至少有一个模态的样本
                if len(sample['modalities']) >= 1:
                    samples.append(sample)
        
        return samples
    
    def _load_nifti(self, path: str) -> np.ndarray:
        """加载NIfTI文件"""
        try:
            nii = nib.load(path)
            data = nii.get_fdata().astype(np.float32)
            return data
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def _apply_modality_dropout(self, available_modalities: List[str]) -> List[str]:
        """
        应用Modality Dropout
        随机丢弃一些模态以增加组合多样性
        """
        if len(available_modalities) <= self.min_modalities:
            return available_modalities
        
        kept_modalities = []
        for mod in available_modalities:
            if random.random() > self.modality_dropout_prob:
                kept_modalities.append(mod)
        
        # 确保至少保留min_modalities个模态
        if len(kept_modalities) < self.min_modalities:
            # 随机选择需要保留的模态
            kept_modalities = random.sample(available_modalities, self.min_modalities)
        
        return kept_modalities
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 检查缓存
        if self.cache_data and idx in self.cache:
            cached_data = self.cache[idx]
            images = cached_data['images'].clone()
            original_observed = cached_data['observed'].clone()
        else:
            # 初始化输出张量
            num_modalities = len(self.MODALITY_ORDER)
            images = torch.zeros(num_modalities, *self.image_size, dtype=torch.float32)
            original_observed = torch.zeros(num_modalities, dtype=torch.float32)
            
            # 加载每个模态
            for i, modality in enumerate(self.MODALITY_ORDER):
                if modality in sample['modalities']:
                    path = sample['modalities'][modality]
                    data = self._load_nifti(path)
                    
                    if data is not None:
                        # 确保数据尺寸正确
                        if data.shape == self.image_size:
                            images[i] = torch.from_numpy(data)
                            original_observed[i] = 1.0
                        else:
                            print(f"Warning: Size mismatch for {path}, expected {self.image_size}, got {data.shape}")
            
            # 缓存数据
            if self.cache_data:
                self.cache[idx] = {
                    'images': images.clone(),
                    'observed': original_observed.clone()
                }
        
        # 不再应用Modality Dropout，直接使用原始observed
        observed = original_observed.clone()
        
        # 应用空间数据增强
        if self.augmentation:
            # 只对observed的模态应用增强
            # 创建TorchIO Subject
            subject_dict = {}
            for i, modality in enumerate(self.MODALITY_ORDER):
                if observed[i] == 1.0:
                    # TorchIO需要4D张量 (C, D, H, W)
                    subject_dict[modality] = tio.ScalarImage(tensor=images[i:i+1])
            
            if subject_dict:
                subject = tio.Subject(**subject_dict)
                transformed = self.spatial_transform(subject)
                
                # 将增强后的数据放回images张量
                for i, modality in enumerate(self.MODALITY_ORDER):
                    if modality in subject_dict:
                        images[i] = transformed[modality].data[0]
        
        return {
            'images': images,  # (num_modalities, D, H, W)
            'observed': observed,  # (num_modalities,)

        }


def create_pretrain_dataloader(
    excel_dir: str = "/home/data/Match_data_path/pretraining_processed",
    batch_size: int = 4,
    num_workers: int = 8,
    augmentation: bool = True,
    modality_dropout_prob: float = 0.3,
    min_modalities: int = 1,
    shuffle: bool = True,
    pin_memory: bool = True,
    cache_data: bool = False,
) -> DataLoader:
    """
    创建预训练数据加载器
    
    Args:
        excel_dir: Excel文件目录
        batch_size: 批量大小
        num_workers: 数据加载进程数
        augmentation: 是否数据增强
        modality_dropout_prob: 模态dropout概率
        min_modalities: 至少保留的模态数
        shuffle: 是否打乱数据
        pin_memory: 是否使用pinned memory
        cache_data: 是否缓存数据到内存
    
    Returns:
        DataLoader实例
    """
    dataset = MultiModalPretrainDataset(
        excel_dir=excel_dir,
        augmentation=augmentation,
        modality_dropout_prob=modality_dropout_prob,
        min_modalities=min_modalities,
        cache_data=cache_data,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    return dataloader


def collate_fn_with_info(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数，处理批量数据
    """
    images = torch.stack([item['images'] for item in batch])
    observed = torch.stack([item['observed'] for item in batch])

    return {
        'images': images,  # (B, num_modalities, D, H, W)
        'observed': observed,  # (B, num_modalities)
    }


# ============== 使用示例 ==============
if __name__ == '__main__':
    print("=" * 60)
    print("多模态3D医学图像预训练数据加载器")
    print("=" * 60)
    
    # 创建数据加载器
    dataloader = create_pretrain_dataloader(
        excel_dir="/home/data/Match_data_path/pretraining_processed",
        batch_size=2,
        num_workers=4,
        augmentation=True,
        modality_dropout_prob=0.3,
        min_modalities=1,
        shuffle=True,
    )
    
    print(f"\n数据集大小: {len(dataloader.dataset)}")
    print(f"批量数: {len(dataloader)}")
    print(f"模态顺序: {MultiModalPretrainDataset.MODALITY_ORDER}")
    
    # 测试加载一个批量
    print("\n测试加载一个批量...")
    for batch in dataloader:
        images = batch['images']
        observed = batch['observed']
        
        print(f"\n批量数据形状:")
        print(f"  images: {images.shape}")  # (B, 4, 128, 128, 128)
        print(f"  observed: {observed.shape}")  # (B, 4)
        
    
    print("\n" + "=" * 60)
    print("数据加载测试完成!")
    print("=" * 60)

