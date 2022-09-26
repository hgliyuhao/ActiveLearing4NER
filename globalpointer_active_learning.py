import numpy as np
import fairies as fa
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
from tqdm import tqdm
import math
# from bert4keras.layers import EfficientGlobalPointer as GlobalPointer

maxlen = 256
epochs = 20
batch_size = 12
learning_rate = 2e-5
categories = set()

p = '/home/pre_models/chinese-roberta-wwm-ext-tf/'
config_path = p + 'bert_config.json'
checkpoint_path = p + 'bert_model.ckpt'
dict_path = p + 'vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def read_data(filename):

    train_data = fa.read_json(filename)

    res = []

    for text in train_data:

        # 数据例子 ["上海睿昂基因科技股份有限公司","职位变动_辞职_公司",[14,28]]

        for tag_data in train_data[text]:
            categories.add(tag_data[1])

        # 转换成globalPointer的数据格式

        new = [text]
        for tag_data in train_data[text]:
            entry = tag_data[0]
            start = text.find(entry)
            end = text.find(entry) + len(entry) - 1
            # 超过截断长度的不参与计算
            # if end < maxlen:
            #     new.append((start, end, tag_data[1]))
            new.append((start, end, tag_data[1]))
        res.append(new)

    return res


# 标注数据
train_data = read_data('data/train.json')
dev_data = read_data('data/dev.json')
test_data = read_data('data/test.json')

categories = list(sorted(categories))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)


model = build_transformer_model(config_path, checkpoint_path)
output = GlobalPointer(len(categories), 64)(model.output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=global_pointer_crossentropy,
    optimizer=Adam(learning_rate),
    metrics=[global_pointer_f1_score])


class NamedEntityRecognizer(object):
    """命名实体识别器
    """

    def recognize(self, text, threshold=0):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append((mapping[start][0], mapping[end][-1],
                             categories[l]))

        scores = scores.clip(-1, 1)

        # LC_score越大，模型对预测的结果信息越低，样本携带的信息越多，越值得被标注
        LC_score = (1 - np.abs(np.prod(scores, axis=2))).sum()
        # LC_score = (np.abs(np.prod(scores, axis=2))).sum()

        return entities, LC_score


NER = NamedEntityRecognizer()


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R, LC_score = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def get_score(data):
    """评测函数
    """

    res = []

    for d in tqdm(data, ncols=100):

        X, Y, Z = 1e-10, 1e-10, 1e-10
        entities, LC_score = NER.recognize(d[0])

        text_len = min(512, len(d[0]))

        R = set(entities)
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

        MNLP_confidence = LC_score * 512 / text_len
        entry_MNLP_confidence = LC_score / (
            (len(R) + 2)**0.5) / 2 * 512 / text_len

        new = {}
        new['text'] = d[0]
        # new["predict_entries"] = list(R)
        # new["entries"] = list(T)
        new["len"] = len(d[0])
        new['f1_socre'] = f1
        new['LC_score'] = float(LC_score)
        new['MNLP_confidence'] = float(MNLP_confidence)
        new['entry_MNLP_confidence'] = float(entry_MNLP_confidence)

        res.append(new)

    fa.write_json("example/globalpointer_confidence.json", res, isIndent=True)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(dev_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('model/globalpointer.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n'
            % (f1, precision, recall, self.best_val_f1))


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator])

    model.load_weights('model/globalpointer.weights')
    get_score(test_data)