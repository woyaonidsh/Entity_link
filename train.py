from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(self, model, criterion, optimizer, datasize, batchsize, device, tokenizer):
        super(Trainer, self).__init__()
        self.batchsize = batchsize  # 梯度累计
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0
        self.datasize = datasize  # 数据集大小
        self.device = device
        self.tokenizer = tokenizer

    # in batch negatives
    def construct_batch(self, dataset):
        data_tensor = TensorDataset(dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5])
        data_loader = DataLoader(dataset=data_tensor, batch_size=self.batchsize, shuffle=True)
        return data_loader

    def loss_function(self, score):
        loss = torch.tensor([0], dtype=torch.float).to(self.device)
        for i in range(len(score)):
            sub_loss = torch.tensor([0], dtype=torch.float).to(self.device)
            sum_loss = torch.tensor([0], dtype=torch.float).to(self.device)
            for j in range(len(score[i])):
                if i == j:
                    sub_loss = score[i][j]
                else:
                    sum_loss += torch.exp(score[i][j])
            loss += (torch.log(sum_loss) - sub_loss)
        loss = loss / self.batchsize
        return loss

    # helper function for training
    def train(self, dataset):
        new_data = self.construct_batch(dataset)
        self.optimizer.zero_grad()
        total_loss = 0.0
        for i, data in tqdm(enumerate(new_data), desc='Training epoch ' + str(self.epoch + 1) + ': '):
            if data[0].size(0) < self.batchsize:
                break
            entity, category, description, text, left, right = data[0].to(self.device), data[1].to(self.device), \
                                                               data[2].to(self.device), data[3].to(self.device), \
                                                               data[4].to(self.device), data[5].to(self.device)
            output = self.model(entity, left, right, text, entity, category, description)  # 模型的输入
            loss = self.loss_function(output)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if i % 2 == 0 and i != 0:
                print('Loss is: %.2f' % (total_loss / i))
        self.epoch += 1
        return total_loss / self.datasize[0]
