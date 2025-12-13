# src/glove_utils.py

import numpy as np


def load_glove_embeddings(glove_path: str, word2idx: dict, embedding_dim: int = 300):
    """
    Load GloVe embeddings from a text file and build an embedding matrix
    that matches our vocabulary (word2idx).

    Parameters
    ----------
    glove_path : str
        Path to the GloVe file, e.g. 'data/glove.840B.300d.txt'.
    word2idx : dict
        Mapping from word string to index in our vocabulary.
    embedding_dim : int
        Dimensionality of the GloVe vectors (300 for glove.840B.300d).

    Returns
    -------
    np.ndarray
        Embedding matrix of shape (vocab_size, embedding_dim).
        Row i corresponds to the vector for the word whose index is i.
        Words not found in GloVe are initialized randomly.
    """
    vocab_size = len(word2idx)

    # Initialize embedding matrix with random normal values.
    # This will be used for words that do not appear in the GloVe file.
    embeddings = np.random.normal(scale=0.6,
                                  size=(vocab_size, embedding_dim)).astype("float32")

    print(f"Start loading GloVe vectors from {glove_path} ...")
    found = 0

    # Read the GloVe file line by line.
    # Each line: word followed by 'embedding_dim' floating point values.
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            # Skip invalid lines
            if len(parts) < embedding_dim + 1:
                continue

            word = parts[0]
            if word in word2idx:
                try:
                    vector = np.asarray(parts[1:], dtype="float32")
                except ValueError:
                    # If conversion to float fails, skip this line
                    continue

                if vector.shape[0] != embedding_dim:
                    # Skip if dimension does not match (should not happen for correct file)
                    continue

                idx = word2idx[word]
                embeddings[idx] = vector
                found += 1

    print(f"GloVe hit {found}/{vocab_size} words.")
    return embeddings
