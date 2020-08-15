import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import numpy as np
from string import punctuation
from collections import Counter

# 常数
train_ratio = 0.9
batch_size = 50
EPOCH = 4
num_test = 2000
num_train = 10000

# 加载并处理文件
with open("./data/reviews.txt", 'r') as f:
    reviews = f.read()
with open("./data/labels.txt", 'r') as f:
    labels = f.read()


def address(text):
    text = text.lower()
    text = "".join([c for c in text if c not in punctuation])  # 删去标点
    review = text.split("\n")  # 以句划分
    text = " ".join(review)
    word = text.split()  # 以词划分
    return review, word


all_reviews, all_words = address(reviews)

# embedding review
counter = Counter(all_words)
word_list = sorted(counter, key=counter.get, reverse=True)
dict_w2i = {word: idx + 1 for idx, word in enumerate(word_list)}  # 从单词到数字的字典
int_to_vocab = {idx: word for word, idx in dict_w2i.items()}  # 从数字到单词的字典
encoded_reviews = []
for _ in all_reviews:
    encoded_reviews += [[dict_w2i[word] for word in _.split()]]

# label 文字转数字
all_labels = labels.split("\n")
encoded_labels = np.array([[1.0 if label == "positive" else 0.0 for label in all_labels]])
encoded_labels = encoded_labels.reshape(25001, 1)


# 统一句子长度为200
def modify_text_length(review, seq_length):
    ret = np.zeros((len(review), seq_length), dtype=int)
    for i, r in enumerate(review):
        for j in range(min(seq_length, len(r))):
            ret[i, j] = review[i][j]
    return ret


reviews_modified = modify_text_length(encoded_reviews, 200)

# 划分training和test的部分
train_x, train_y = reviews_modified[:num_train], encoded_labels[:num_train]
test_x, test_y = reviews_modified[num_train:num_train+num_test], encoded_labels[num_train:num_train+num_test]

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# LSTM模块
class LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # [0~dic_w2i] => [100]
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                           bidirectional=True, batch_first= True, dropout=0.5)  # [100] => [256]
        self.fc = nn.Linear(hidden_dim * 2, 1)  # [256*2] => [1]
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        # [seq, b, 1] => [seq, b, 100]
        output, (hidden, cell) = self.rnn(embedding)
        # output: [seq, b, hid_dim*2]
        # hidden/cell: [num_layers*2, b, hid_dim]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        # [b, hid_dim*2] => [b, 1]
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out


rnn = LSTM(len(dict_w2i), 100, 256)
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
loss_func = nn.BCEWithLogitsLoss()


def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(rnn, iterator, optimizer, criteon):
    avg_acc = []
    rnn.train()

    for i, (batch_x, batch_y) in enumerate(iterator):

        # [seq, b] => [b, 1] => [b]
        pred = rnn(batch_x)
        #
        loss = criteon(pred, batch_y)
        acc = binary_acc(pred, batch_y).item()
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(i, acc)

    avg_acc = np.array(avg_acc).mean()
    print('avg acc:', avg_acc)


def eval(rnn, iterator, criteon):
    avg_acc = []

    rnn.eval()

    with torch.no_grad():
        for b_x, b_y in iterator:
            # [b, 1] => [b]
            pred = rnn(b_x)
            loss = criteon(pred, b_y)
            acc = binary_acc(pred, b_y).item()
            avg_acc.append(acc)

    avg_acc = np.array(avg_acc).mean()

    print('>>test:', avg_acc)

for epoch in range(10):
    eval(rnn, test_loader, loss_func)
    train(rnn, train_loader, optimizer, loss_func)