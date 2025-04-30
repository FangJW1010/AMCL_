import torch
import torch.nn as nn
import torch.nn.functional as F
import math
##########################################################
class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, filter_sizes: list, output_dim: int,
                 dropout: float, feature_dim: int = 128):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])

        hidden_dim = len(filter_sizes) * n_filters


        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 10, hidden_dim * 5),
            nn.Mish(),
            nn.Dropout(),
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, feature_dim)
        )


        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()

    def forward(self,data,return_features=False,length=None):
        embedded = self.embedding(data)
        embedded = embedded.permute(0, 2, 1)
        conved = [self.Mish(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]
        flatten = [pool.view(pool.size(0), -1) for pool in pooled]
        cat = self.dropout(torch.cat(flatten, dim=1))


        features = self.feature_extractor(cat)


        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits
#################################






