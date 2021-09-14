import torch
import torch.nn as nn
import config
import json
from pytorch_transformers import BertTokenizer
from tqdm import tqdm
from model import entity_bi_encoder, Encoder_entity
from torch.utils.data import TensorDataset, DataLoader

arg = config.parse_args()


def padding_token(dataset, length):
    new_data = []
    for i in dataset:
        if len(i) > length:
            new_data.append(i[0:length])
        else:
            news = i
            for j in range(0, length - len(i)):
                news.append(0)
            new_data.append(news)
    return new_data


def batch_data(entity_token, category_token, description_token):
    entity_token = padding_token(entity_token, 10)
    category_token = padding_token(category_token, 10)
    description_token = padding_token(description_token, 500)
    # tensor数据
    entity_tensor = torch.tensor(entity_token)
    category_tensor = torch.tensor(category_token)
    description_tensor = torch.tensor(description_token)
    return [entity_tensor, category_tensor, description_tensor]


def entity_encode(arg):
    # 训练设备
    device = torch.device("cuda:0" if arg.cuda else "cpu")  # 查看是否具有GPU
    print('*' * 80)
    print('The current device: ', device)  # 输出当前设备名

    # 构建模型
    mention_model = entity_bi_encoder.Span_encoder(num_layer=arg.num_layer, embed_dim=arg.embed_dim,
                                                   vocabsize=arg.vocab,
                                                   in_feature=arg.in_feature, out_feature=arg.out_feature)
    mention_con_model = entity_bi_encoder.Mention_context_encoder(lstm_hidden_dim=arg.lstm_dim, vocabsize=arg.vocab,
                                                                  embed_dim=arg.embed_dim, batchsize=arg.batch_size,
                                                                  lstm_num_layers=arg.lstm_layers, dropout=arg.dropout)
    entity_model = entity_bi_encoder.Span_encoder(num_layer=arg.num_layer, embed_dim=arg.embed_dim, vocabsize=arg.vocab,
                                                  in_feature=arg.in_feature, out_feature=arg.out_feature)
    entity_con_model = entity_bi_encoder.entity_context_encoder(lstm_hidden_dim=arg.lstm_dim, vocabsize=arg.vocab,
                                                                embed_dim=arg.embed_dim, batchsize=arg.batch_size,
                                                                lstm_num_layers=arg.lstm_layers,
                                                                num_layer=arg.num_layer,
                                                                in_feature=arg.in_feature, out_feature=arg.out_feature)
    en_com_model = entity_bi_encoder.entity_encoder(out_feature=arg.out_feature, d_model=arg.d_model, nhead=arg.nhead,
                                                    dim_feedforward=arg.dim_feed, num_layer=arg.tran_layer)
    men_com_model = entity_bi_encoder.mention_encoder(out_feature=arg.out_feature, d_model=arg.d_model, nhead=arg.nhead,
                                                      dim_feedforward=arg.dim_feed, num_layer=arg.tran_layer)
    model = Encoder_entity.encoder(mention=mention_model, mention_context=mention_con_model, entity=entity_model,
                                   entity_context=entity_con_model, entity_encoder=en_com_model,
                                   mention_encoder=men_com_model)
    model = nn.DataParallel(model).cuda()

    # 加载模型参数
    model.load_state_dict(torch.load(arg.model_file))

    tokenizer = BertTokenizer.from_pretrained(arg.token_path)

    # 加载实体集
    candidates = []
    entity = []
    category = []
    description = []
    with open(arg.save_data, 'r', encoding='utf-8') as entity_text:
        for datas in entity_text:
            data = json.loads(datas)
            candidates.append(data['entity'])
            entity.append(data['entity'])
            category.append(data['category'])
            description.append(data['description'])

    # tokenize
    entity_token = []
    for i in tqdm(entity, desc='Tokenize entity: '):
        entity_token.append(tokenizer.encode(i))
    category_token = []
    for i in tqdm(category, desc='Tokenize category: '):
        category_token.append(tokenizer.encode(i))
    description_token = []
    for i in tqdm(description, desc='Tokenize description: '):
        description_token.append(tokenizer.encode(i))

    # tensor
    dataset = batch_data(entity_token, category_token, description_token)

    # 构建数据集
    data_tensor = TensorDataset(dataset[0], dataset[1], dataset[2])
    data_batch = DataLoader(dataset=data_tensor, batch_size=arg.test_batch, shuffle=False)

    # 计算候选实体
    with torch.no_grad():
        # 计算实体库实体的特征向量
        entity_feature = []
        for i, en_data in tqdm(enumerate(data_batch), desc='Encoder entity feature: '):
            an_entity = en_data[0].cuda()
            an_category = en_data[1].cuda()
            an_description = en_data[2].cuda()
            y = model(entity=an_entity, category=an_category, entity_context=an_description)
            entity_feature.append(y.cpu())
        entity_feature = torch.cat(entity_feature, dim=0)
        print('The size of entity feature: ', entity_feature.shape)
        torch.save(entity_feature, arg.entity_feature)
        return entity_feature.transpose(-1, -2), candidates