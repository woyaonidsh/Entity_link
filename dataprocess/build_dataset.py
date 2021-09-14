import json
from tqdm import tqdm
import config

arg = config.parse_args()


def load_entity_pro(file_path, save_file, save_related):
    entity = []
    category = []
    description = []
    related = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for data in tqdm(file.readlines(), desc='Extract entity property: '):
            dataset = data.split('\t')
            if '简介' in dataset:
                entity.append(dataset[1])
                category.append(dataset[2])
                description.append(dataset[4])
            if '相关实体' in dataset:
                related.append([dataset[1], dataset[4]])
    file.close()
    print('The number of entity: ', len(entity))
    with open(save_file, 'w', encoding='utf-8') as save:
        for i in range(len(entity)):
            dic = {'entity': entity[i], 'category': category[i], 'description': description[i]}
            dic = json.dumps(dic, ensure_ascii=False)
            save.write(dic)
            save.write('\n')
    with open(save_related, 'w', encoding='utf-8') as relate:
        for i in related:
            dic = {'entity': i[0], 'related_entity': i[1]}
            dic = json.dumps(dic, ensure_ascii=False)
            relate.write(dic)
            relate.write('\n')
    save.close()
    relate.close()


# 判断entity是否在text中
def entity_in_text(entity, text, save_file):
    entitys = []  # entity
    texts = []  # 包含entity的文本
    for i in tqdm(entity, desc='entity in text: '):
        for j in text:
            if i in j:
                entitys.append(i)
                texts.append(j)
                break
    print('The number of entity_text: ', len(entitys))
    with open(save_file, 'w', encoding='utf-8') as save:
        for i in range(len(entitys)):
            dic = {'entity': entitys[i], 'text': texts[i]}
            dic = json.dumps(dic, ensure_ascii=False)
            save.write(dic)
            save.write('\n')
    save.close()


# 加载数据集
def load_data(file_path, entity_path, save_file):
    entity_data = []
    # 读取entity
    with open(entity_path, 'r', encoding='utf-8') as entity_file:
        for data in entity_file.readlines():
            entity_data.append(data.split('\t')[1])
    entity_data = list(set(entity_data))
    print('The number of entity: ', len(entity_data))
    # 保存实体集
    with open(save_file, 'w', encoding='utf-8') as save_entity:
        number = [_ for _ in range(0, len(entity_data))]
        save = dict(zip(number, entity_data))
        saves = json.dumps(save, ensure_ascii=False)
        save_entity.write(saves)
    save_entity.close()
    all_txt = []
    src_txt = []
    tgt_txt = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for data in tqdm(file.readlines(), desc='Load data: '):
            text = json.loads(data)
            tgt_txt.append(text['tgt_text'])
            src_txt.append(text['src_text'])
            all_txt.append(text['tgt_text'] + text['src_text'])
    print('The number of text: ', len(all_txt))
    file.close()
    return entity_data, all_txt, tgt_txt, src_txt


def process_entity_text(entity_base, text_base, related_base, save_file):
    entity_pro = []
    # 加载entity属性
    with open(entity_base, 'r', encoding='utf-8') as entity_data:
        for datas in entity_data:
            data = json.loads(datas)
            entity_pro.append([data['entity'], data['category'], data['description']])
    entity_data.close()

    # 加载entity和text数据
    men_text = []
    with open(text_base, 'r', encoding='utf-8') as text_data:
        for datas in text_data:
            data = json.loads(datas)
            men_text.append([data['entity'], data['text']])
    text_data.close()

    # 加载相关实体
    related = []
    with open(related_base, 'r', encoding='utf-8') as related_data:
        for datas in related_data:
            data = json.loads(datas)
            related.append([data['entity'], data['related_entity']])
    related_data.close()

    dataset = []
    for i in tqdm(entity_pro, desc='check entity in text: '):
        for j in men_text:
            if i[0] == j[0]:
                dataset.append([i[0], i[1], i[2], j[-1]])
    for i in tqdm(dataset, desc='check entity in related: '):
        for j in related:
            if i[0] == j[0]:
                i.append(j[1])
            elif i[0] == j[1]:
                i.append(j[0])

    with open(save_file, 'w', encoding='utf-8') as file:
        for i in dataset:
            dic = {'entity': i[0], 'category': i[1], 'description': i[2], 'text': i[3], 'related_entity': i[4:],
                   'most_related': i[0]}
            dic = json.dumps(dic, ensure_ascii=False)
            file.write(dic)
            file.write('\n')
        file.close()
    print('The number of dataset: ', len(dataset))

# 取出实体集相关属性
# load_entity_pro(arg.entity_pro_path, arg.save_entity, arg.save_related)
# 从文本中取出数据
# entity_data, all_txt, tgt_txt, src_txt = load_data(arg.text_path, arg.entity_path, arg.save_entity_num)
# 将包含实体的文本与实体配对，都成训练集
# entity_in_text(entity_data, all_txt, arg.save_en_men)
# process_entity_text(arg.save_entity, arg.save_en_men, arg.save_related, arg.save_data)
