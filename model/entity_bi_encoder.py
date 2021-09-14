import torch
import torch.nn as nn
import math


class BiLSTM(nn.Module):
    def __init__(self, lstm_hidden_dim, vocabsize, embed_dim, batchsize, num_layer=3, lstm_num_layers=2, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.batch_size = batchsize
        self.embeding = nn.Embedding(num_embeddings=vocabsize, embedding_dim=embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, lstm_hidden_dim // 2, num_layers=lstm_num_layers, dropout=dropout,
                              bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim)

    def forward(self, x):
        bilstm_out, (h, c) = self.bilstm(self.embeding(x))  # 得到bilstm的输出
        bilstm_out = self.layer_norm(bilstm_out)
        return bilstm_out  # dimension:[batchsize * 1700 * hidden_dim]


class Mention_context_encoder(nn.Module):
    def __init__(self, lstm_hidden_dim, vocabsize, embed_dim, batchsize, lstm_num_layers=2, dropout=0.1):
        super(Mention_context_encoder, self).__init__()
        self.left_context = BiLSTM(lstm_hidden_dim, vocabsize, embed_dim, batchsize, lstm_num_layers)
        self.right_context = BiLSTM(lstm_hidden_dim, vocabsize, embed_dim, batchsize, lstm_num_layers)
        self.sentence_text = BiLSTM(lstm_hidden_dim, vocabsize, embed_dim, batchsize, lstm_num_layers)

    def forward(self, left_context, right_context, sentence_context):
        left = self.left_context(left_context)
        right = self.right_context(right_context)
        sentence = self.sentence_text(sentence_context)
        return left, right, sentence


class Span_encoder(nn.Module):
    def __init__(self, num_layer, embed_dim, vocabsize, in_feature, out_feature):
        super(Span_encoder, self).__init__()
        self.span = nn.ModuleList()
        self.embeding = nn.Embedding(num_embeddings=vocabsize, embedding_dim=embed_dim, padding_idx=0)
        self.layer = nn.Linear(embed_dim, in_feature)
        self.span.append(nn.Linear(in_feature, 1024))
        self.span.append(nn.Linear(1024, 2048))
        self.span.append(nn.Linear(2048, out_feature))
        self.relu = nn.ReLU()

    def forward(self, span_context):
        x = self.embeding(span_context)
        x = self.layer(x)
        for layer in self.span:
            x = layer(x)
            x = self.relu(x)
        return x  # 维度 [batch size * 10 * out_feature]


class entity_encoder(nn.Module):
    def __init__(self, out_feature, d_model, nhead, dim_feedforward, num_layer):
        super(entity_encoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                            dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=num_layer,
                                             norm=nn.LayerNorm(d_model))
        self.layer = nn.Linear(out_feature, d_model)

    def forward(self, entity, category, description):
        x = torch.cat([entity, category, description], dim=1)
        x = self.layer(x)
        x = self.encoder(x)
        return x


class mention_encoder(nn.Module):
    def __init__(self, out_feature, d_model, nhead, dim_feedforward, num_layer):
        super(mention_encoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                            dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=num_layer,
                                             norm=nn.LayerNorm(d_model))
        self.layer = nn.Linear(out_feature, d_model)

    def forward(self, mention, left, right, text):
        x = torch.cat([mention, left, right, text], dim=1)
        x = self.layer(x)
        x = self.encoder(x)
        return x


class entity_context_encoder(nn.Module):
    def __init__(self, lstm_hidden_dim, vocabsize, embed_dim, batchsize, lstm_num_layers, num_layer, in_feature,
                 out_feature):
        super(entity_context_encoder, self).__init__()
        self.sentence = BiLSTM(lstm_hidden_dim, vocabsize, embed_dim, batchsize, lstm_num_layers)
        self.category = nn.ModuleList()
        self.embding = nn.Embedding(num_embeddings=vocabsize, embedding_dim=embed_dim, padding_idx=0)

        self.layer = nn.Linear(embed_dim, in_feature)
        self.category.append(nn.Linear(in_feature, 512))
        self.category.append(nn.Linear(512, 1024))
        self.category.append(nn.Linear(1024, out_feature))
        self.relu = nn.ReLU()

    def forward(self, context, category):
        x = self.embding(category)
        x = self.layer(x)
        for layer in self.category:
            x = layer(x)
            x = self.relu(x)
        context = self.sentence(context)
        return x, context
