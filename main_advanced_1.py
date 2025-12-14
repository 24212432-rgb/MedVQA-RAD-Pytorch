# main_advanced.py
import os
import torch
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

# Introduce your configuration and modules
from src import config
from src.dataset_advanced import VQARADSeqDataset
from src.model_advanced import VQAModelAdvanced
from src.train_advanced_1 import train_model_seq


def main():
    # 1. Initialization Tokenizer
    print("Initialize BERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 2. Definition of nuclear power data enhancement (Data Augmentation)
    # -----------------------------------------------------------------
    print("Setting up Augmentation Strategy (Targeting >70% Acc)...")

    # Training set: Increase the difficulty level to prevent rote learning.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Rotate left and right
        transforms.RandomRotation(degrees=15),  # Randomly rotate by +/- 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Lighting changes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Test set: Keep it as is, only perform standardization
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # -----------------------------------------------------------------

    # 3. Load the dataset
    # Technique: Instantiate the Dataset twice, passing in different transforms each time.
    print("Loading Datasets...")

    # Full data loader (for splitting)
    full_dataset_train_mode = VQARADSeqDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        tokenizer=tokenizer,
        transform=train_transform,  # Training mode: With enhancement
        max_q_len=config.MAX_Q_LEN,
        max_a_len=config.MAX_A_LEN
    )

    full_dataset_test_mode = VQARADSeqDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        tokenizer=tokenizer,
        transform=test_transform,  # Test mode: No enhancement
        max_q_len=config.MAX_Q_LEN,
        max_a_len=config.MAX_A_LEN
    )

    # 4. Manually divide the training/test sets (80% / 20%)
    dataset_size = len(full_dataset_train_mode)
    indices = list(range(dataset_size))

    # Fix the random seed to ensure reproducibility
    random.seed(42)
    random.shuffle(indices)

    split = int(0.8 * dataset_size)
    train_indices = indices[:split]
    test_indices = indices[split:]

    # Create the final dataset using Subset
    # The training set uses full_dataset_train_mode (with augmentation)
    train_dataset = torch.utils.data.Subset(full_dataset_train_mode, train_indices)
    # The test set uses full_dataset_test_mode (without augmentation)
    test_dataset = torch.utils.data.Subset(full_dataset_test_mode, test_indices)

    print(f"Train Size: {len(train_dataset)} | Test Size: {len(test_dataset)}")

    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 6. Initialize the model
    print("Building Model (BERT + ResNet)...")
    model = VQAModelAdvanced(
        vocab_size=tokenizer.vocab_size,  #The vocabulary size of BERT (~30522)
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout_p=0.3  # Adding Dropout to prevent overfitting after enhancement
    )

    # 7. Start training (transfer control to train_advanced.py)
    train_model_seq(
        model,
        train_loader,
        test_loader,
        config=config,
        tokenizer=tokenizer
    )


if __name__ == "__main__":

    main()
