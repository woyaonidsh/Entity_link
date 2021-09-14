import torch
import config
import torch.nn as nn
import logging
from dataprocess import build_dataset, process
from pytorch_transformers import BertTokenizer
from model import BERT_encoder, bert
import train
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

arg = config.parse_args()


def main(arg):
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 训练设备
    print('*' * 80)

    # 加载数据集
    tokenizer = BertTokenizer.from_pretrained(arg.token_path)

    entity_token, category_token, description_token, left_token, right_token = \
        process.load_data(arg.save_entity, tokenizer)

    dataset = process.batch_data(entity_token, category_token, description_token, left_token,
                                 right_token)
    data_size = len(dataset[0])

    # 构建模型
    mention_model = BERT_encoder.Span_encoder(embed_dim=arg.embed_dim, in_feature=arg.in_feature,
                                              out_feature=arg.out_feature)
    mention_con_model = BERT_encoder.Mention_context_encoder(d_model=arg.d_model, nhead=arg.nhead,
                                                             dim_feedforward=arg.dim_feedforward,
                                                             num_layer=arg.tran_layer)
    entity_model = BERT_encoder.Span_encoder(embed_dim=arg.embed_dim, in_feature=arg.in_feature,
                                             out_feature=arg.out_feature)
    category_model = BERT_encoder.Span_encoder(embed_dim=arg.embed_dim, in_feature=arg.in_feature,
                                               out_feature=arg.out_feature)
    entity_con_model = BERT_encoder.entity_context_encoder(d_model=arg.d_model, nhead=arg.nhead,
                                                           dim_feedforward=arg.dim_feedforward,
                                                           num_layer=arg.tran_layer)
    en_com_model = BERT_encoder.entity_encoder(d_model=arg.d_model, nhead=arg.nhead,
                                               dim_feedforward=arg.dim_feedforward,
                                               num_layer=arg.trans_layer)
    men_com_model = BERT_encoder.mention_encoder(d_model=arg.d_model, nhead=arg.nhead,
                                                 dim_feedforward=arg.dim_feedforward,
                                                 num_layer=arg.trans_layer)
    model = bert.encoder(mention=mention_model, mention_context=mention_con_model, entity=entity_model,
                         entity_context=entity_con_model, entity_encoder=en_com_model,
                         mention_encoder=men_com_model, category=category_model, vocab=arg.vocab,
                         embed_dim=arg.embed_dim)
    model = nn.DataParallel(model)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()  # Loss function
    optimizer = torch.optim.SGD(params=model.parameters(), lr=arg.lr, momentum=0.9)  # optimize

    # 构建训练器
    trainer = train.Trainer(model=model, criterion=criterion, optimizer=optimizer, datasize=data_size,
                            batchsize=arg.batch_size, tokenizer=tokenizer)

    # 开始训练模型
    for epoch in range(arg.epochs):
        train_loss = trainer.train(dataset)
        print('Epoch ', str(epoch))
        print('Epoch Loss is: %.2f' % (train_loss))

    # 保存模型参数
    torch.save(model.state_dict(), arg.model_file)


if __name__ == "__main__":
    main(arg=arg)
