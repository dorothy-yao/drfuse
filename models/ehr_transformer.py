import torch
from torch import nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.rand(1, max_len, d_model))
        self.pe.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # x: (batch_size, seq_len, embedding_dim)
        return self.dropout(x)


class EHRTransformer(nn.Module):
    def __init__(self, input_size, num_classes,
                 d_model=256, n_head=8, n_layers_feat=1,
                 n_layers_shared=1, n_layers_distinct=1,
                 dropout=0.3, max_len=350):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.emb = nn.Linear(input_size, d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=0, max_len=max_len)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_feat = nn.TransformerEncoder(layer, num_layers=n_layers_feat)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_shared = nn.TransformerEncoder(layer, num_layers=n_layers_shared)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_distinct = nn.TransformerEncoder(layer, num_layers=n_layers_distinct)
        self.fc_distinct = nn.Linear(d_model, num_classes)

    def forward(self, x, seq_lengths):
        attn_mask = torch.stack([torch.cat([torch.zeros(len_, device=x.device),
                                 float('-inf')*torch.ones(max(seq_lengths)-len_, device=x.device)])
                                for len_ in seq_lengths])
        x = self.emb(x) # * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        feat = self.model_feat(x, src_key_padding_mask=attn_mask)
        h_shared = self.model_shared(feat, src_key_padding_mask=attn_mask)
        h_distinct = self.model_distinct(feat, src_key_padding_mask=attn_mask)

        padding_mask = torch.ones_like(attn_mask).unsqueeze(2)
        padding_mask[attn_mask==float('-inf')] = 0
        rep_shared = (padding_mask * h_shared).sum(dim=1) / padding_mask.sum(dim=1)
        rep_distinct = (padding_mask * h_distinct).sum(dim=1) / padding_mask.sum(dim=1)

        pred_distinct = self.fc_distinct(rep_distinct).sigmoid()

        return rep_shared, rep_distinct, pred_distinct
