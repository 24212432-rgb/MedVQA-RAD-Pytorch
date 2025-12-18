import os

# Obtain the current directory where the file is located and the root directory of the project
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)

# Data and file path configuration
DATA_JSON_PATH = os.path.join(BASE_DIR, "data", "VQA_RAD Dataset Public.json")
IMG_DIR_PATH   = os.path.join(BASE_DIR, "data", "VQA_RAD Image Folder")
GLOVE_PATH     = os.path.join(BASE_DIR, "data", "glove.840B.300d.txt")  # If no word vector file is provided, it can be ignored.

# Model and training hyperparameter configuration
BATCH_SIZE    = 8      # Batch size can be adjusted according to the available memory.
NUM_EPOCHS    = 30     # Number of training rounds
LEARNING_RATE = 1e-3   # Learning rate
MAX_Q_LEN     = 30     # Maximum length of the question sequence (truncated if exceeded)
MAX_A_LEN     = 30     # Maximum length of answer sequence (for advanced models, including <bos> and <eos>)
EMBED_DIM     = 300    # Word vector dimension (consistent with GloVe)
HIDDEN_DIM    = 512    # The dimension of the LSTM hidden layer


