import torch
import torch.nn as nn
import config

arg = config.parse_args()


class encoder(nn.Module):
    def __init__(self, mention, mention_context, entity, entity_context, entity_encoder, mention_encoder):
        super(encoder, self).__init__()
        self.mention = mention
        self.mention_context = mention_context
        self.entity = entity
        self.entity_context = entity_context
        self.entity_encoder = entity_encoder
        self.mention_encoder = mention_encoder

    def forward(self, mention, left_context, right_context, sentence):
        left, right, sen = self.mention_context(left_context, right_context, sentence)
        men = self.mention(mention)
        x = self.mention_encoder(men, left, right, sen)
        # 维度 [batch size * 25 * d_model]
        x = x[:, 0:1:].squeeze(1)

        return x

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
