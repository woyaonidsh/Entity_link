## Introduction  
The project is mainly about entity link technology. The input of the task is a piece of text and the corresponding entity, and the output is the related entity set of the entity in the entity database. I use a two-tower and hierarchical neural network to solve this problem

## Model
The model consists of entity-encoder and mention-encoder. Entity-encoder is used to model entity and description of entity. Mention-encoder is used to model mention and
context of mention. Specially, I split context into left context and right context so that this model can capture more information about mention.  
 ![image](https://github.com/woyaonidsh/Entity_link/blob/main/model.png)  

## Resource
The data set and the trained model were uploaded to the online disk. the [data set and trained model](https://pan.baidu.com/s/1eN1ACCW3-pQvAAsjQvGZUA)&nbsp;&nbsp;&nbsp;&nbsp;passward : 2021  

## Environments
`python 3.7`  
`cuda 10.2`  
`pytorch 1.7.1+cu101`  

## Run
The fist step is to download data set and use the following introduction:  
`python main.py`  

## Test
Download the data set and trained model. Then, use the following introduction:  
`python test.py`  

## Attention
I used the `BertTokenizer` to change words into tokens, so you need to prepare `Bert` in advance. The uploaded data set is also processed and I don't upload raw data in online disk.  

## Contact
If you have any `questions` or want to give me some `suggestions`, please contact with me, I will response it as soon as possible.
My Email：jzl1601763588@163.com or 1120182394@bit.edu.cn
