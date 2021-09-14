import torch
import torch.nn as nn
import config
import json
from pytorch_transformers import BertTokenizer
from tqdm import tqdm
from model import mention_bi_encoder, Encoder_mention
import compute_entity

arg = config.parse_args()


# 得到context
def get_left_right(text, entity):
    left_text = []
    right_text = []
    for i in tqdm(range(len(entity)), desc='Split left and right text: '):
        number = text[i].split(entity[i])
        left_text.append(''.join(number[: len(number) // 2]) + entity[i])
        right_text.append(entity[i] + ''.join(number[len(number) // 2:]))
    return left_text, right_text


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


def batch_data(mention_token, left_token, right_token, text_token):
    mention_token = padding_token(mention_token, 10)
    left_token = padding_token(left_token, 500)
    right_token = padding_token(right_token, 500)
    text_token = padding_token(text_token, 3000)
    # tensor数据
    mention_tensor = torch.tensor(mention_token)
    left_tensor = torch.tensor(left_token)
    right_tensor = torch.tensor(right_token)
    text_tensor = torch.tensor(text_token)
    return [mention_tensor, left_tensor, right_tensor, text_tensor]


def main(arg):
    # 训练设备
    device = torch.device("cuda:0" if arg.cuda else "cpu")  # 查看是否具有GPU
    print('*' * 80)
    print('The current device: ', device)  # 输出当前设备名

    # 构建模型
    mention_model = mention_bi_encoder.Span_encoder(num_layer=arg.num_layer, embed_dim=arg.embed_dim,
                                                    vocabsize=arg.vocab,
                                                    in_feature=arg.in_feature, out_feature=arg.out_feature)
    mention_con_model = mention_bi_encoder.Mention_context_encoder(lstm_hidden_dim=arg.lstm_dim, vocabsize=arg.vocab,
                                                                   embed_dim=arg.embed_dim, batchsize=arg.batch_size,
                                                                   lstm_num_layers=arg.lstm_layers, dropout=arg.dropout)
    entity_model = mention_bi_encoder.Span_encoder(num_layer=arg.num_layer, embed_dim=arg.embed_dim,
                                                   vocabsize=arg.vocab,
                                                   in_feature=arg.in_feature, out_feature=arg.out_feature)
    entity_con_model = mention_bi_encoder.entity_context_encoder(lstm_hidden_dim=arg.lstm_dim, vocabsize=arg.vocab,
                                                                 embed_dim=arg.embed_dim, batchsize=arg.batch_size,
                                                                 lstm_num_layers=arg.lstm_layers,
                                                                 num_layer=arg.num_layer,
                                                                 in_feature=arg.in_feature, out_feature=arg.out_feature)
    en_com_model = mention_bi_encoder.entity_encoder(out_feature=arg.out_feature, d_model=arg.d_model, nhead=arg.nhead,
                                                     dim_feedforward=arg.dim_feed, num_layer=arg.tran_layer)
    men_com_model = mention_bi_encoder.mention_encoder(out_feature=arg.out_feature, d_model=arg.d_model,
                                                       nhead=arg.nhead,
                                                       dim_feedforward=arg.dim_feed, num_layer=arg.tran_layer)
    model = Encoder_mention.encoder(mention=mention_model, mention_context=mention_con_model, entity=entity_model,
                                    entity_context=entity_con_model, entity_encoder=en_com_model,
                                    mention_encoder=men_com_model)
    model = nn.DataParallel(model).cuda()

    # 加载模型参数
    model.load_state_dict(torch.load(arg.model_file))

    tokenizer = BertTokenizer.from_pretrained(arg.token_path)
    test_text = []
    test_entity = []

    # 加载测试数据
    with open(arg.test_file, 'r', encoding='utf-8') as test:
        for data in test:
            data_text = json.loads(data)
            test_text.append(data_text['text'])
            test_entity.append(data_text['entity'])

    # 得到context
    left_text, right_text = get_left_right(test_text, test_entity)

    test_t_token = []
    for i in tqdm(test_text, desc='Tokenize test text: '):
        test_t_token.append(tokenizer.encode(i))
    test_e_token = []
    for i in tqdm(test_entity, desc='Tokenize test entity: '):
        test_e_token.append(tokenizer.encode(i))
    test_l_token = []
    for i in tqdm(left_text, desc='Tokenize test left text: '):
        test_l_token.append(tokenizer.encode(i))
    test_r_token = []
    for i in tqdm(right_text, desc='Tokenize test right text: '):
        test_r_token.append(tokenizer.encode(i))

    # 构建测试集
    dataset = batch_data(mention_token=test_e_token, left_token=test_l_token, right_token=test_r_token,
                         text_token=test_t_token)

    # 计算当前实体的特征
    with torch.no_grad():
        mention_feature = []
        for i in tqdm(range(len(dataset[3])), desc='Encoder mention feature: '):
            an_mention = dataset[0][i].unsqueeze(0).cuda()
            an_left = dataset[1][i].unsqueeze(0).cuda()
            an_right = dataset[2][i].unsqueeze(0).cuda()
            an_text = dataset[3][i].unsqueeze(0).cuda()
            x = model(mention=an_mention, left_context=an_left, right_context=an_right, sentence=an_text)
            mention_feature.append(x.cpu())

    # 计算实体库中所有实体的特征
    entity_feature, candidates = compute_entity.entity_encode(arg)

    score_result = []
    # 计算得分
    for i in mention_feature:
        score = torch.mm(i, entity_feature).squeeze(0).cpu()
        score_result.append(score)

    candidate_entity = []

    # 计算最相关实体
    ss = []
    for i in tqdm(score_result, desc='compute the related entity: '):
        candidate = []
        for j in range(len(i)):
            candidate.append((-i[j], candidates[j]))
        result = sorted(candidate, key=lambda x: (x[0]), reverse=True)
        candidate_entity.append(result[0:10])
        ss.append(result)

    for i in range(len(test_entity)):
        print(test_entity[i], '的相关实体: ', candidate_entity[i])


"""
    for i in range(len(test_entity)):
        jishu = 0
        for j in ss[i]:
            jishu += 1
            if test_entity[i] == j[-1]:
                print(j)
                print('排位: ', jishu)
"""

if __name__ == "__main__":
    main(arg=arg)
