# src/model_advanced.py
import torch
import torch.nn as nn
import torchvision.models as models


class VQAModelAdvanced(nn.Module):
    """
    Advanced seq2seq VQA model with Attention and Dropout support.
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 300,
            hidden_dim: int = 512,
            dropout_p: float = 0.3,  # <--- 新增参数：接收 main.py 传来的 0.3
            embedding_matrix=None,
            train_cnn: bool = False,
            max_dec_steps: int = 30,
    ):
        super(VQAModelAdvanced, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_dec_steps = max_dec_steps

        # 1. Image Encoder (ResNet50)
        # ----------------------------------------------------------------
        try:
            from torchvision.models import ResNet50_Weights
            resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception:
            resnet = models.resnet50(pretrained=True)

        # 移除 avgpool 和 fc，保留空间特征 (B, 2048, 7, 7)
        modules = list(resnet.children())[:-2]
        self.resnet_features = nn.Sequential(*modules)

        # 冻结 CNN 权重 (防止小数据集过拟合)
        for param in self.resnet_features.parameters():
            param.requires_grad = train_cnn

        # 图像特征映射: 2048 -> hidden_dim
        self.v_proj = nn.Linear(2048, hidden_dim)

        # 2. Question Encoder (Embedding + LSTM)
        # ----------------------------------------------------------------
        # 注意：虽然用的是 BERT Tokenizer 的 ID，但为了简化训练，我们这里重头训练 Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 问题编码器 LSTM
        self.enc_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # 3. Answer Decoder (Attention + LSTM)
        # ----------------------------------------------------------------
        self.dec_lstm = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim)

        # Attention Layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_linear = nn.Linear(hidden_dim, 1)

        # Output Layer
        self.out_linear = nn.Linear(hidden_dim, vocab_size)

        # Dropout (关键！防止数据增强后过拟合)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, images, questions, decoder_input):
        """
        Args:
            images: [B, 3, 224, 224]
            questions: [B, Q_Len]
            decoder_input: [B, A_Len] (Shifted targets)
        """
        batch_size = images.size(0)

        # --- A. Image Encoding ---
        # [B, 2048, 7, 7]
        img_feats = self.resnet_features(images)
        # Flatten -> [B, 2048, 49] -> [B, 49, 2048]
        img_feats = img_feats.view(batch_size, 2048, -1).permute(0, 2, 1)
        # Project -> [B, 49, hidden_dim]
        img_feats = self.v_proj(img_feats)
        img_feats = torch.tanh(img_feats)
        img_feats = self.dropout(img_feats)  # Dropout

        # --- B. Question Encoding ---
        # [B, Q_Len, Embed]
        q_embeds = self.embedding(questions)
        q_embeds = self.dropout(q_embeds)  # Dropout

        # Encoder Output: _, (h_n, c_n)
        _, (h_enc, c_enc) = self.enc_lstm(q_embeds)

        # Initial Decoder State = Encoder Final State
        h_dec = h_enc.squeeze(0)  # [B, Hidden]
        c_dec = c_enc.squeeze(0)

        # --- C. Decoding Loop ---
        seq_len = decoder_input.size(1)
        outputs = []

        # Embedding inputs for decoder
        dec_embeds = self.embedding(decoder_input)  # [B, A_Len, Embed]
        dec_embeds = self.dropout(dec_embeds)  # Dropout

        # Attention Project for Query (h_dec changes, but let's pre-calc part if needed)
        # Here we do it step-by-step

        for t in range(seq_len):
            # 1. Calculate Attention
            # Query: h_dec [B, Hidden] -> [B, 1, Hidden]
            h_dec_expanded = self.q_proj(h_dec).unsqueeze(1)

            # Score: tanh(Image + Query) -> [B, 49, Hidden]
            attn_energy = torch.tanh(img_feats + h_dec_expanded)

            # Alpha: [B, 49, 1]
            alpha = torch.softmax(self.attn_linear(attn_energy), dim=1)

            # Context: Sum(alpha * Image) -> [B, Hidden]
            context = (alpha * img_feats).sum(dim=1)

            # 2. LSTM Step
            # Input: [Word_Embed_t, Context]
            lstm_input = torch.cat([dec_embeds[:, t, :], context], dim=1)

            h_dec, c_dec = self.dec_lstm(lstm_input, (h_dec, c_dec))

            # 3. Output Prediction
            logits = self.out_linear(self.dropout(h_dec))  # [B, Vocab]
            outputs.append(logits.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, A_Len, Vocab]

    def generate_answer(self, images, questions, bos_idx, eos_idx):
        """
        Greedy decoding for inference.
        """
        batch_size = images.size(0)
        device = images.device

        # --- Image ---
        img_feats = self.resnet_features(images)
        img_feats = img_feats.view(batch_size, 2048, -1).permute(0, 2, 1)
        img_feats = self.v_proj(img_feats)
        img_feats = torch.tanh(img_feats)

        # --- Question ---
        q_embeds = self.embedding(questions)
        _, (h_enc, c_enc) = self.enc_lstm(q_embeds)
        h_dec = h_enc.squeeze(0)
        c_dec = c_enc.squeeze(0)

        # --- Decoding ---
        # Start token
        curr_token = torch.full((batch_size,), bos_idx, dtype=torch.long, device=device)

        generated_seqs = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(self.max_dec_steps):
            # Embed current token
            curr_emb = self.embedding(curr_token)  # [B, Embed]

            # Attention
            h_dec_expanded = self.q_proj(h_dec).unsqueeze(1)
            attn_energy = torch.tanh(img_feats + h_dec_expanded)
            alpha = torch.softmax(self.attn_linear(attn_energy), dim=1)
            context = (alpha * img_feats).sum(dim=1)

            # LSTM Step
            lstm_input = torch.cat([curr_emb, context], dim=1)
            h_dec, c_dec = self.dec_lstm(lstm_input, (h_dec, c_dec))

            # Predict
            logits = self.out_linear(h_dec)
            _, best_idx = torch.max(logits, dim=1)  # [B]

            curr_token = best_idx  # Auto-regressive

            for i in range(batch_size):
                if not finished[i]:
                    token = best_idx[i].item()
                    if token == eos_idx:
                        finished[i] = True
                    else:
                        generated_seqs[i].append(token)

            if all(finished):
                break

        return generated_seqs