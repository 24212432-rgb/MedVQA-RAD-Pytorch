import os

# 获取当前文件所在目录和项目根目录
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)

# 数据与文件路径配置
DATA_JSON_PATH = os.path.join(BASE_DIR, "data", "VQA_RAD Dataset Public.json")
IMG_DIR_PATH   = os.path.join(BASE_DIR, "data", "VQA_RAD Image Folder")
GLOVE_PATH     = os.path.join(BASE_DIR, "data", "glove.840B.300d.txt")  # 如果未提供词向量文件，可忽略

# 模型和训练超参数配置
BATCH_SIZE    = 8      # 批大小，根据内存可调整
NUM_EPOCHS    = 30     # 训练轮数
LEARNING_RATE = 1e-3   # 学习率
MAX_Q_LEN     = 30     # 问题序列最大长度（超过则截断）
MAX_A_LEN     = 30     # 答案序列最大长度（用于高级模型，包含<bos>/<eos>）
EMBED_DIM     = 300    # 词向量维度（与GloVe一致）
HIDDEN_DIM    = 512    # LSTM隐层维度
