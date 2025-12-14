# main_advanced.py
import os
import torch
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

# 引入你的配置和模块
from src import config
from src.dataset_advanced import VQARADSeqDataset
from src.model_advanced import VQAModelAdvanced
from src.train_advanced_1 import train_model_seq


def main():
    # 1. 初始化 Tokenizer
    print("Initialize BERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 2. 定义核动力数据增强 (Data Augmentation)
    # -----------------------------------------------------------------
    print("Setting up Augmentation Strategy (Targeting >70% Acc)...")

    # 训练集：增加难度，防止死记硬背
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # 左右翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转 +/- 15度
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 光照变化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 测试集：保持原样，只做标准化
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # -----------------------------------------------------------------

    # 3. 加载数据集
    # 技巧：实例化两次 Dataset，分别传入不同的 transform
    print("Loading Datasets...")

    # 全量数据加载器 (用于切分)
    full_dataset_train_mode = VQARADSeqDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        tokenizer=tokenizer,
        transform=train_transform,  # 训练模式：带增强
        max_q_len=config.MAX_Q_LEN,
        max_a_len=config.MAX_A_LEN
    )

    full_dataset_test_mode = VQARADSeqDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        tokenizer=tokenizer,
        transform=test_transform,  # 测试模式：无增强
        max_q_len=config.MAX_Q_LEN,
        max_a_len=config.MAX_A_LEN
    )

    # 4. 手动划分训练/测试集 (80% / 20%)
    dataset_size = len(full_dataset_train_mode)
    indices = list(range(dataset_size))

    # 固定随机种子以保证可复现性
    random.seed(42)
    random.shuffle(indices)

    split = int(0.8 * dataset_size)
    train_indices = indices[:split]
    test_indices = indices[split:]

    # 使用 Subset 创建最终数据集
    # 训练集用 full_dataset_train_mode (带增强)
    train_dataset = torch.utils.data.Subset(full_dataset_train_mode, train_indices)
    # 测试集用 full_dataset_test_mode (无增强)
    test_dataset = torch.utils.data.Subset(full_dataset_test_mode, test_indices)

    print(f"Train Size: {len(train_dataset)} | Test Size: {len(test_dataset)}")

    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 6. 初始化模型
    print("Building Model (BERT + ResNet)...")
    model = VQAModelAdvanced(
        vocab_size=tokenizer.vocab_size,  # BERT的词表大小 (~30522)
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout_p=0.3  # 增加Dropout防止增强后的过拟合
    )

    # 7. 开始训练 (移交控制权给 train_advanced.py)
    train_model_seq(
        model,
        train_loader,
        test_loader,
        config=config,
        tokenizer=tokenizer
    )


if __name__ == "__main__":
    main()