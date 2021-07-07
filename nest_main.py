import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import fairies as fa
from bert4keras.backend import keras, search_layer, K
import math
# keras_version == 0.10.0

maxlen = 256
batch_size = 16
categories = set()

def read_data(filename):

    train_data = fa.read_json(filename)
    
    res = []
    
    for r in train_data:

        text = r

        # 数据例子 ["上海睿昂基因科技股份有限公司","职位变动_辞职_公司",[14,28]]

        for tag_data in train_data[r]:
            categories.add(tag_data[1])

        res.append([text,train_data[r]])
    
    return res        

a = read_data('train.json')
categories = list(sorted(categories))

fa.write_json('categories.json',categories)

categories.insert(0,'i')
categories.insert(1,'o')
print(categories)



num_labels = len(categories)
id2label,label2id = fa.label2id(categories)

p = 'D:/lyh/model/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
config_path = p +'bert_config.json'
checkpoint_path = p + 'bert_model.ckpt'
dict_path = p +'vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def search(pattern, sequence):

    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """

    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class data_generator(DataGenerator):

    """数据生成器

    """

    def __iter__(self, random=False):

        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, result in self.sample(random):

            text = result[0]
            predicts = result[1]

            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            seq_len = len(token_ids)
            labels = [[0] * num_labels for i in range(seq_len)]

            for predict in predicts:

                #["上海睿昂基因科技股份有限公司","职位变动_辞职_公司",[14,28]]
                entry = predict[0]
                entry_type = predict[1]
                position  = predict[2]

                entry_token_ids = tokenizer.encode(entry)[0][1:-1]
                entry_start = search(entry_token_ids, token_ids)

                if entry_start != -1:
                    entry_type_index = label2id[entry_type]
                    labels[entry_start][entry_type_index] = 1 

                    for i in range(1,len(entry_token_ids)):
                        labels[entry_start + i][1] = 1
                                  
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False
    )

output = Dense(units=num_labels,
               activation='sigmoid',
               kernel_initializer=model.initializer)(model.output)
model = Model(model.input, output)
model.summary()


train_generator = data_generator(a, batch_size)

def extract_arguments(text):
    
    # 等你真的到了这里 你才能懂这里的风景
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 510:
        tokens.pop(-2)

    mapping = tokenizer.rematch(text, tokens)
    
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    labels = model.predict([[token_ids], [segment_ids]])[0]
    
    labels = labels[1:]

    for lable in labels:
        for i in range(len(lable)):
            if lable[i] >= 0.4:
                lable[i] = 1
            else:
                lable[i] = 0 

    res = find_entry(labels,mapping,text)
    return res

def find_entry(labels,mapping,text):

    res = []

    for k,label in enumerate(labels):
        for i,l in enumerate(label):
            if l == 1 and i != 1:
                start_type = id2label[i]
                start = k
                end = 0
                j = k + 1
                while j < len(labels) and labels[j][1] == 1:
                    end = j
                    j += 1
                if end > start:
                    if len(mapping[end+1]) > 0:
                        entry = text[mapping[start+1][0]:mapping[end+1][-1] +1]
                        res.append([entry,start_type])
    return res

def compute_LC(text):

    # 计算预测中概率最大的预测序列的概率值

    tokens = tokenizer.tokenize(text)
    while len(tokens) > 510:
        tokens.pop(-2)

    mapping = tokenizer.rematch(text, tokens)
    
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    labels = model.predict([[token_ids], [segment_ids]])[0]
    
    labels = labels[1:]

    confidence = 0

    for lable in labels:
        con = 1
        for l in lable:
            if l <= 0.5:
                l = 1 - l
            con *= l
        confidence += con   
    print(confidence)
    return confidence

def compute_MNLP(text):

    # 计算预测中概率最大的预测序列的概率值

    tokens = tokenizer.tokenize(text)
    while len(tokens) > 510:
        tokens.pop(-2)

    mapping = tokenizer.rematch(text, tokens)
    
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    labels = model.predict([[token_ids], [segment_ids]])[0]
    
    labels = labels[1:]

    confidence = 0

    for lable in labels:
        con = 1
        for l in lable:
            if l <= 0.5:
                l = 1 - l
            con += math.log(l)
        confidence += con   
    return (confidence/len(labels))  

def predict_for_tag(text):
    
    # 等你真的到了这里 你才能懂这里的风景
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 510:
        tokens.pop(-2)

    mapping = tokenizer.rematch(text, tokens)
    
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    labels = model.predict([[token_ids], [segment_ids]])[0]
    
    labels = labels[1:]

    lc_confidence = 0
    MNLP_confidence = 0

    for lable in labels:
        lc_con = 1
        mnlp_con = 1
        for l in lable:
            if l <= 0.5:
                l = 1 - l
            lc_con *= l
            mnlp_con += math.log(l)
        lc_confidence += lc_con   
        MNLP_confidence += mnlp_con

    MNLP_confidence = MNLP_confidence/(len(labels))
    # entry_MNLP_confidence = 1 - (1 - MNLP_confidence)/((len(res) + 2)**0.5) * (2*0.5)
    for lable in labels:
        for i in range(len(lable)):
            if lable[i] >= 0.4:
                lable[i] = 1
            else:
                lable[i] = 0 

    res = find_entry(labels,mapping,text)

    new = {}

    new['res'] = res
    new['LC'] = lc_confidence
    new['MNLP_confidence'] = MNLP_confidence
    new['entry_MNLP_confidence'] = 1 - (1 - MNLP_confidence)/((len(res) + 2)**0.5) * (2*0.5)


    return new

@fa.clock
def evaluate(filename):
    
    # 评估函数
    D = fa.read_json(filename)
    X, Y, Z = 1, 1, 1 
    
    for i in D:
        text = i
        T = extract_arguments(text) 
        dev_list = D[i]
        R = []
        for j in dev_list:
            R.append([j[0],j[1]])

        same = 0
        for i in R:
            if i in T:
                same += 1

        X += same
        Y += len(R)
        Z += len(T)

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    print(f1)
    print(precision)
    print(recall)
    return f1, precision, recall         

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('last_model.weights')
        val_acc,precision, recall = evaluate('dev.json')
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        print(val_acc)
        print(self.best_val_acc)

        fa.print_to_log(val_acc)
        fa.print_to_log(self.best_val_acc)
   
evaluator = Evaluator()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),
    metrics=['accuracy']
)

model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=140,
    callbacks=[evaluator]
)

model.load_weights('best_model.weights')

# model.load_weights('10x_0.748_best_model.weights')
# text = "记者 | 杨霞\n编辑｜殷幼安\n界面新闻记者从多位知情人士处独家获悉，中通快递近日收购了海尔集团旗下的日日顺乐家智能快递柜。\n据知情人士透露，这一收购主要涉及日日顺乐家在全国铺设的约3万组智能快递柜。中通快递网点将以6000-7000元/组的价格认购当地的日日顺智能柜，并承担未来的场租、运营成本，这些快递柜可以承接中通及其他同行的快递包裹，为用户提供快速投放、自助取件服务。\n界面新闻记者分别向双方求证此事，中通快递方面表示暂不回应，海尔日日顺截至发稿前仍未回复。\n公开资料显示，日日顺乐家智能快递柜由海尔集团旗下日日顺研发，于2014年12月宣布在全国200座城市同步启动入驻，2015年在全国286座城市投放了1万多组社区智能柜和社区驿站平台，覆盖用户1800万。2016年，乐家曾计划在全国范围内进驻4万个社区，到2017年要覆盖全国8万个社区。\n但是，智能快递柜行业普遍面临成本高、盈利模式单一的问题，日日顺乐家快递柜在2019年也曾因超时收费陷入争议。而且，日日顺旗下的物流服务专注于家电等大件物流配送业务，这与快递柜之间的业务协同性并不大。\n目前，日日顺乐家业务重心转向搭建一个围绕家庭、社区、城市的综合服务平台，强化家电、净水、家居、采暖、充电桩等业务"
# res = extract_arguments(text)
# print(res)

predict_path = r'D:\lyh\AL_NER\data\公告.json'

predict_data = fa.read_json(predict_path)

predict_data = list(set(predict_data))

# text = predict_data[1]
# print(text)

new = {}

for i in predict_data:
    res = predict_for_tag(i)
    new[i] = res

fa.write_json('duplicate_removal_res.json',new)    


# evaluate('dev.json')

