import torch
import json
from tqdm import tqdm


def statistic_data(dataset, name):
    length = []
    for i in dataset:
        length.append(len(i))
    print('The average length of ' + name + ': ', sum(length) // len(length))
    print('The Max length of ' + name + ': ', max(length))
    print('The min length of ' + name + ': ', min(length))


def load_data(data_path, tokenizer):
    entity = []
    category = []
    description = []
    left_text = []
    right_text = []
    with open(data_path, 'r', encoding='utf-8') as datas:
        for data in datas:
            txt = json.loads(data)
            entity.append(txt['entity'])
            category.append(txt['category'])
            description.append(txt['description'])
    for i in tqdm(range(len(entity)), desc='Split left and right text: '):
        number = description[i].split(entity[i])
        left_text.append(''.join(number[: len(number) // 2]) + entity[i])
        right_text.append(entity[i] + ''.join(number[len(number) // 2:]))
    entity_token = []
    for i in tqdm(entity, desc='Tokenize entity: '):
        entity_token.append(tokenizer.encode(i))
    category_token = []
    for i in tqdm(category, desc='Tokenize category: '):
        category_token.append(tokenizer.encode(i))
    description_token = []
    for i in tqdm(description, desc='Tokenize description: '):
        description_token.append(tokenizer.encode(i))
    left_token = []
    for i in tqdm(left_text, desc='Tokenize left text: '):
        left_token.append(tokenizer.encode(i))
    right_token = []
    for i in tqdm(right_text, desc='Tokenize right text: '):
        right_token.append(tokenizer.encode(i))
    statistic_data(entity_token, 'entity')
    statistic_data(category_token, 'category')
    statistic_data(description_token, 'description')
    statistic_data(left_token, 'left_text')
    statistic_data(right_token, 'right_text')
    return entity_token, category_token, description_token, left_token, right_token


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


def batch_data(entity_token, category_token, description_token, left_token, right_token):
    entity_token = padding_token(entity_token, 10)
    category_token = padding_token(category_token, 10)
    description_token = padding_token(description_token, 500)
    left_token = padding_token(left_token, 250)
    right_token = padding_token(right_token, 250)
    # tensor数据
    entity_tensor = torch.tensor(entity_token)
    category_tensor = torch.tensor(category_token)
    description_tensor = torch.tensor(description_token)
    left_tensor = torch.tensor(left_token)
    right_tensor = torch.tensor(right_token)
    return [entity_tensor, category_tensor, description_tensor, left_tensor, right_tensor]


"""
entity_token, category_token, description_token, text_token, left_token, right_token = load_data(arg.save_data)

entity_tensor, category_tensor, description_tensor, text_tensor, left_tensor, right_tensor = \
     batch_data(entity_token, category_token, description_token, text_token, left_token, right_token)
"""

"""
batchs = []
    for i in range(len(entity_token)):
        shu = random.randint(0, len(entity_token), batch_size)
        entity = [entity_token[i]]
        category = [category_token[i]]
        description = [description_token[i]]
        text = [text_token[i]]
        left = [left_token[i]]
        right = [right_token[i]]
        for j in shu:
            entity.append(entity_token[j])
            category.append(category_token[j])
            text.append(text_token[j])
            description.append(description_token[j])
            left.append(left_token[j])
            right.append(right_token[j])
        entity_tensor = torch.tensor(entity)
        category_tensor = torch.tensor(category)
        description_tensor = torch.tensor(description)
        text_tensor = torch.tensor(text)
        left_tensor = torch.tensor(left)
        right_tensor = torch.tensor(right)
        batchs.append([entity_tensor, category_tensor, description_tensor, text_tensor, left_tensor, right_tensor])
"""