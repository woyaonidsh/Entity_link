import torch
import torch.nn as nn
import config

arg = config.parse_args()


class encoder(nn.Module):
    def __init__(self, mention, mention_context, entity, category, entity_context, entity_encoder, mention_encoder,
                 vocab, embed_dim):
        super(encoder, self).__init__()
        self.mention = mention
        self.mention_context = mention_context
        self.entity = entity
        self.entity_context = entity_context
        self.entity_encoder = entity_encoder
        self.mention_encoder = mention_encoder
        self.category = category
        self.embeding = nn.Embedding(num_embeddings=vocab, embedding_dim=embed_dim, padding_idx=0)

    def forward(self, mention, left_context, right_context, sentence, entity, category, entity_context):
        mention = self.embeding(mention)
        left_context = self.embeding(left_context)
        right_context = self.embeding(right_context)
        sentence = self.embeding(sentence)
        entity = self.embeding(entity)
        category = self.embeding(category)
        entity_context = self.embeding(entity_context)

        # mention
        sen = self.mention_context(left_context, right_context, sentence)
        men = self.mention(mention)
        x = self.mention_encoder(men, sen)
        # 维度 [batch size * 25 * d_model]
        x = x[:, 0:1:].squeeze(1)

        # entity
        des = self.entity_context(entity_context)
        cate = self.category(category)
        enti = self.entity(entity)
        # 维度 [batch size * 25 * d_model]
        y = self.entity_encoder(enti, cate, des)
        y = y[:, 0:1:].squeeze(1)

        return x, y

    def embed_entity(self, entity, category, entity_context):
        cate, des = self.entity_context(entity_context, category)
        enti = self.entity(entity)
        y = self.entity_encoder(enti, cate, des)
        y = self.layer(y.transpose(-1, -2)).squeeze(-1).transpose(-1, -2)
        return y

    def embed_mention(self, mention, left_context, right_context, sentence):
        left, right, sen = self.mention_context(left_context, right_context, sentence)
        men = self.mention(mention)
        x = self.mention_encoder(men, left, right, sen)
        x = self.layer(x.transpose(-1, -2)).squeeze(-1)
        return x

    def score_ranker(self, mention, entity):
        score = torch.mm(mention, entity)
        return score


"""
mention_model = bi_encoder.Span_encoder(num_layer=arg.num_layer, embed_dim=arg.embed_dim, vocabsize=arg.vocab,
                                        in_feature=arg.in_feature, out_feature=arg.out_feature)
mention_con_model = bi_encoder.Mention_context_encoder(lstm_hidden_dim=arg.lstm_dim, vocabsize=arg.vocab,
                                                       embed_dim=arg.embed_dim, batchsize=arg.batch_size,
                                                       lstm_num_layers=arg.lstm_layers, dropout=arg.dropout)
entity_model = bi_encoder.Span_encoder(num_layer=arg.num_layer, embed_dim=arg.embed_dim, vocabsize=arg.vocab,
                                       in_feature=arg.in_feature, out_feature=arg.out_feature)
entity_con_model = bi_encoder.entity_context_encoder(lstm_hidden_dim=arg.lstm_dim, vocabsize=arg.vocab,
                                                     embed_dim=arg.embed_dim, batchsize=arg.batch_size,
                                                     lstm_num_layers=arg.lstm_layers, num_layer=arg.num_layer,
                                                     in_feature=arg.in_feature, out_feature=arg.out_feature)
en_com_model = bi_encoder.entity_encoder(out_feature=arg.out_feature, d_model=arg.d_model, nhead=arg.nhead,
                                         dim_feedforward=arg.dim_feed, num_layer=arg.tran_layer)
men_com_model = bi_encoder.mention_encoder(out_feature=arg.out_feature, d_model=arg.d_model, nhead=arg.nhead,
                                           dim_feedforward=arg.dim_feed, num_layer=arg.tran_layer)
model = encoder(mention=mention_model, mention_context=mention_con_model, entity=entity_model,
                entity_context=entity_con_model, entity_encoder=en_com_model, mention_encoder=men_com_model).cuda()

print(model)

entity = torch.ones([16, 10], dtype=torch.int64).cuda()
category = torch.ones([16, 10], dtype=torch.int64).cuda()
description = torch.ones([16, 500], dtype=torch.int64).cuda()
text = torch.ones([16, 3500], dtype=torch.int64).cuda()
left_text = torch.ones([16, 1700], dtype=torch.int64).cuda()
right_text = torch.ones([16, 1700], dtype=torch.int64).cuda()

out = model(entity, left_text, right_text, text, entity, category, description)
print(out)

"""
"""
entity = torch.ones([32, 10], dtype=torch.int64)
category = torch.ones([32, 10], dtype=torch.int64)
description = torch.ones([32, 500], dtype=torch.int64)
text = torch.ones([32, 3500], dtype=torch.int64)
left_text = torch.ones([32, 1700], dtype=torch.int64)
right_text = torch.ones([32, 1700], dtype=torch.int64)

out3 = mention_model(entity)
out = mention_con_model(left_text, right_text, text)

out1, out2 = entity_con_model(description, category)

outt = en_com_model(out3, out1, out2)

print(outt.shape)
"""

