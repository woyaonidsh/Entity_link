import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    # TODO ---------------dataset------------------  datasets arguments
    parser.add_argument('--entity_path', type=str, default='D:\Datesets\\blink\data\\military.txt')
    parser.add_argument('--text_path', type=str, default='D:\Datesets\\blink\data\\train_merge_final.txt')
    parser.add_argument('--entity_pro_path', type=str, default='D:\Datesets\\blink\data\\military_prop.txt')
    parser.add_argument('--token_path', type=str, default='D:\Homework\pretrained_model\chinese_L-12_H-768_A-12')
    parser.add_argument('--save_entity', type=str, default='data/entity_base.json')
    parser.add_argument('--save_entity_num', type=str, default='data/entity_num.json')
    parser.add_argument('--save_en_men', type=str, default='data/entity_mention.json')
    parser.add_argument('--save_related', type=str, default='data/related_entity.json')
    parser.add_argument('--save_data', type=str, default='data/dataset.json')

    # TODO ---------------model-------------------- model arguments
    parser.add_argument('--lstm_dim', type=int, default=300)
    parser.add_argument('--vocab', type=int, default=30524)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--in_feature', type=int, default=300)
    parser.add_argument('--out_feature', type=int, default=300)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dim_feed', type=int, default=2048)
    parser.add_argument('--tran_layer', type=int, default=6)

    # TODO ---------------train-------------------- training arguments
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--save_log', type=str, default='checkpoints/', help='')  # 保存训练信息的文件夹
    parser.add_argument('--log_file', type=str, default='train.txt', help='')  # 保存训练信息的文件名
    parser.add_argument('--model_file', type=str, default='checkpoint/model.pth', help='')  # 保存模型文件名
    parser.add_argument('--test_file', type=str, default='data/test.json')
    parser.add_argument('--test_batch', type=int, default=32)
    parser.add_argument('--entity_feature', type=str, default='checkpoint/entity_feature.pth')

    args = parser.parse_args()
    return args
