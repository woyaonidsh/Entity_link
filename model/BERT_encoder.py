import torch
import torch.nn as nn
import math


class BiLSTM(nn.Module):
    def __init__(self, lstm_hidden_dim, lstm_in_feature, lstm_num_layers=1):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(lstm_in_feature, lstm_hidden_dim, num_layers=lstm_num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim)

    def forward(self, x):
        bilstm_out, (h, c) = self.bilstm(x)  # 得到bilstm的输出
        h = self.layer_norm(h).transpose(1, 0)
        return h  # dimension:[batchsize * 1700 * hidden_dim]


class Span_encoder(nn.Module):
    def __init__(self, embed_dim, in_feature, out_feature):
        super(Span_encoder, self).__init__()
        self.span = nn.ModuleList()
        self.layer = nn.Linear(embed_dim, in_feature)
        self.span.append(nn.Linear(in_feature, 512))
        self.span.append(nn.Linear(512, 1024))
        self.span.append(nn.Linear(1024, out_feature))
        self.relu = nn.ReLU()

    def forward(self, span_context):
        x = self.layer(span_context)
        for layer in self.span:
            x = layer(x)
            x = self.relu(x)
        return x  # 维度 [batch size * 10 * out_feature]


class Mention_context_encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layer):
        super(Mention_context_encoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                            dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=num_layer,
                                             norm=nn.LayerNorm(d_model))

    def forward(self, left_context, right_context, sentence_context):
        x = torch.cat([sentence_context, left_context, right_context], dim=1)
        x = self.encoder(x)
        return x


class entity_context_encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layer):
        super(entity_context_encoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                            dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=num_layer,
                                             norm=nn.LayerNorm(d_model))

    def forward(self, context):
        context = self.encoder(context)
        return context


class entity_encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layer):
        super(entity_encoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                            dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=num_layer,
                                             norm=nn.LayerNorm(d_model))

    def forward(self, entity, category, description):
        x = torch.cat([entity, category, description], dim=1)
        x = self.encoder(x)
        return x


class mention_encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layer):
        super(mention_encoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                            dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=num_layer,
                                             norm=nn.LayerNorm(d_model))

    def forward(self, mention, sentence):
        x = torch.cat([mention, sentence], dim=1)
        x = self.encoder(x)
        return x