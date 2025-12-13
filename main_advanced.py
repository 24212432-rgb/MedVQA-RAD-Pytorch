# main_advanced.py
import os
import torch
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from src import config
from src.dataset_advanced import VQARADSeqDataset
from src.model_advanced import VQAModelAdvanced
from src.train_advanced import train_model_seq

def main():
    print("Initialize BERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("Setting up Augmentation Strategy...")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading Datasets...")
    

    full_dataset_train_mode = VQARADSeqDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        tokenizer=tokenizer,
        transform=train_transform,
        only_open=False
    )
    

    full_dataset_test_mode = VQARADSeqDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        tokenizer=tokenizer,
        transform=test_transform,
        only_open=False
    )


    
    train_loader = DataLoader(full_dataset_train_mode, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    

    # Here, we randomly selected 200 samples for verification.
    test_loader = DataLoader(full_dataset_test_mode, batch_size=1, shuffle=False, num_workers=0)

    print(f"Open-Only Train Size: {len(full_dataset_train_mode)}")

    print("Building Model...")
    model = VQAModelAdvanced(
        vocab_size=tokenizer.vocab_size, 
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout_p=0.3 
    )

    train_model_seq(model, train_loader, test_loader, config=config, tokenizer=tokenizer)

if __name__ == "__main__":
    main()