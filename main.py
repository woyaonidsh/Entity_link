import torch
import config
import torch.nn as nn
import logging
from dataprocess import build_dataset, process_data
from pytorch_transformers import BertTokenizer
from model import bi_encoder, Encoder
import train

arg = config.parse_args()


def main(arg):
    # 训练设备
    device = torch.device("cuda:0" if arg.cuda else "cpu")  # 查看是否具有GPU
    print('*' * 80)
    print('The current device: ', device)  # 输出当前设备名

    # 加载数据集
    tokenizer = BertTokenizer.from_pretrained(arg.token_path)

    entity_token, category_token, description_token, text_token, left_token, right_token = \
        process_data.load_data(arg.save_data, tokenizer)

    dataset = process_data.batch_data(entity_token, category_token, description_token, text_token, left_token,
                                      right_token)
    data_size = len(dataset[0])

    # 构建模型
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
    model = Encoder.encoder(mention=mention_model, mention_context=mention_con_model, entity=entity_model,
                            entity_context=entity_con_model, entity_encoder=en_com_model,
                            mention_encoder=men_com_model).to(device)

    criterion = nn.CrossEntropyLoss().to(device)  # Loss function
    optimizer = torch.optim.SGD(params=model.parameters(), lr=arg.lr, momentum=0.9)  # optimize

    # 构建训练器
    trainer = train.Trainer(model=model, criterion=criterion, optimizer=optimizer, datasize=data_size,
                            batchsize=arg.batch_size, device=device, tokenizer=tokenizer)

    # 开始训练模型
    for epoch in range(arg.epochs):
        train_loss = trainer.train(dataset)

    # 保存模型文件
    torch.save(model.state_dict(), arg.model_file)

    # 保存模型参数


if __name__ == "__main__":
    main(arg=arg)
