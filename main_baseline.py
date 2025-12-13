import os
from torch.utils.data import DataLoader
from src import config
from src.dataset import VQARADDataset
from src.model_baseline import VQAModel
from src.train_baseline import train_model
from src.glove_utils import load_glove_embeddings

def main():
    # --- 1. Build train and test datasets ---
    print("Loading VQA-RAD train and test splits ...")

    train_dataset = VQARADDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        split="train",
        max_q_len=config.MAX_Q_LEN,
    )

    # reuse the same vocab and answer mapping for the test set
    test_dataset = VQARADDataset(
        json_path=config.DATA_JSON_PATH,
        img_dir=config.IMG_DIR_PATH,
        split="test",
        word2idx=train_dataset.word2idx,
        answer2idx=train_dataset.answer2idx,
        max_q_len=config.MAX_Q_LEN,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --- 2. Try to load GloVe embeddings (glove.840B.300d.txt) ---
    embedding_matrix = None
    if config.GLOVE_PATH and os.path.exists(config.GLOVE_PATH):
        try:
            print(f"Loading GloVe embeddings from: {config.GLOVE_PATH}")
            embedding_matrix = load_glove_embeddings(
                glove_path=config.GLOVE_PATH,
                word2idx=train_dataset.word2idx,
                embedding_dim=config.EMBED_DIM,
            )
            print(f"GloVe loaded: matrix shape = {embedding_matrix.shape}")
        except Exception as e:
            print(f"[Warning] Failed to load GloVe embeddings: {e}")
            print("         Falling back to random-initialized word embeddings.")
            embedding_matrix = None
    else:
        print(f"[Info] GloVe file not found at: {config.GLOVE_PATH}")
        print("       Using randomly initialized word embeddings instead.")

    # --- 3. Initialize the CNN+LSTM baseline model ---
    model = VQAModel(
        vocab_size=len(train_dataset.word2idx),
        num_answers=len(train_dataset.answer2idx),
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        embedding_matrix=embedding_matrix,
        train_cnn=False,  # keep CNN frozen for CPU training
    )

    # --- 4. Train the baseline model ---
    train_model(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
