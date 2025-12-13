import torch
import torch.nn as nn
import torchvision.models as models

class VQAModel(nn.Module):
    def __init__(self, vocab_size, num_answers, embed_dim=300, hidden_dim=512,
                 embedding_matrix=None, train_cnn=False):
        super(VQAModel, self).__init__()
        # Image encoder: ResNet50 (pre-trained)
        try:
            # Try to use the pre-trained weight interface of the new version of torchvision
            from torchvision.models import ResNet50_Weights
            self.cnn = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception:
            # Compatible with older version interfaces
            self.cnn = models.resnet50(pretrained=True)
        # Obtain the input dimension of the last layer of ResNet and remove its fully connected layer
        num_feats = self.cnn.fc.in_features  # The input dimension of the fc layer in ResNet50 is 2048.
        self.cnn.fc = nn.Identity()         # Replace the fc layer with an identity mapping and output the image feature vector
        # Freeze the parameters of the CNN to accelerate the training (if fine-tuning is needed, set train_cnn to True)
        for param in self.cnn.parameters():
            param.requires_grad = bool(train_cnn)
        # Text encoder: Word embedding + single-layer LSTM
        if embedding_matrix is not None:
            # Initialize the Embedding using pre-trained word vectors
            weight = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
            embed_dim = weight.shape[1]  # Automatically obtain the embedding dimension based on the pre-trained matrix
        else:
            # Randomly initialize the Embedding matrix
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim,
                             num_layers=1, batch_first=True)
        # Integration and Classifier
        self.bn = nn.BatchNorm1d(num_feats + hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(num_feats + hidden_dim, 512)
        self.fc2 = nn.Linear(512, num_answers)

    def forward(self, images, questions):
        # Extract image features [B, 2048]
        img_feats = self.cnn(images)
        # Extract text features [B, hidden_dim]
        embeds = self.embedding(questions)      # [B, T, embed_dim]
        _, (h_n, _) = self.lstm(embeds)         # h_n: [num_layers, B, hidden_dim]
        ques_feat = h_n[-1]                     # The hidden state of the last layer of LSTM [B, hidden_dim]
        # Integrate features and classify
        combined = torch.cat([img_feats, ques_feat], dim=1)  # [B, 2048+512]
        combined = self.bn(combined)
        combined = torch.relu(combined)
        combined = self.dropout(combined)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        logits = self.fc2(x)  # [B, num_answers] The classification scores without Softmax
        return logits
