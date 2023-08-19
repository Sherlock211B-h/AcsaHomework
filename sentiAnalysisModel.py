'''
author: Mingwei Zhang
email: 13337649640@163.com
'''

import torch
import torch.nn as nn
import flask
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from torch import optim
from torch.nn import functional
import pandas as pd
import jieba
import pickle
from tqdm import tqdm
from collections import defaultdict

class Vocab:
    def __init__(self, tokens):
        self.index2token = list()
        self.token2index = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + "<unk>"
            for token in tokens:
                self.index2token.append(token)
                self.token2index[token] = len(self.index2token) - 1
            self.unk = self.token2index["<unk>"]

    @classmethod
    def build_vocab(cls, data, stop_words_file_name, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        stopWords = open(stop_words_file_name, encoding='gb18030', errors='ignore').read().split('\n')
        for i in tqdm(range(data.shape[0]), desc=f"Building vocab"):
            for token in jieba.lcut(data.iloc[i]["review"]):
                if token in stopWords:
                    continue
                token_freqs[token] += 1
        # statistics token frequency
        uniqTokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniqTokens += [token for token, freq in token_freqs.items() \
            if freq >= min_freq and token != "<unk>"]
        return cls(uniqTokens)

    def __len__(self):
        return len(self.index2token)

    def __getitem__(self, token):
        return self.token2index.get(token, self.unk)

    def tokens2ids(self, tokens):
        return [self[token] for token in tokens]

    def ids2tokens(self, ids):
        return [self.index2token[index] for index in ids]


class DataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        embeds = self.embedding(inputs)
        x_pack = pack_padded_sequence(embeds, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(x_pack)
        outputs = self.output(hn[-1])
        log_probs = functional.log_softmax(outputs, dim = -1)
        return log_probs


class SAModel:
    def __init__(self):
        self.data_file_name = "data/data.csv"
        self.stop_file_name = "data/hit_stopwords.txt"
        self.model_file_name = "model/senti_model.pth"
        self.vocab_file_name = "model/vocab.pkl"
        self.embedding_dim = 128
        self.hidden_dim = 24
        self.batch_size = 1024
        self.num_epoch = 10
        self.num_class = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_vocab_loaded = False
        self.is_model_loaded = False

    def collate_fn(self, examples):
        lengths = torch.tensor([len(ex[0]) for ex in examples])
        inputs = [torch.tensor(ex[0]) for ex in examples]
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        inputs = pad_sequence(inputs, batch_first=True)
        return inputs, lengths, targets

    def load_model(self):
        with open(self.vocab_file_name, 'rb') as file:
            self.vocab = pickle.load(file)
        model = LSTM(len(self.vocab), self.embedding_dim, self.hidden_dim, self.num_class)
        model.to(self.device)
        model.load_state_dict(torch.load(self.model_file_name))
        model.eval()
        self.model = model

    def train(self):
        whole_data = pd.read_csv(self.data_file_name, encoding='utf-8')
        self.vocab = Vocab.build_vocab(whole_data, self.stop_file_name)
        self.train_data = [(self.vocab.tokens2ids(sentence), 1) for sentence in
                           whole_data[whole_data["label"] == 1][:20000]["review"]] \
                          + [(self.vocab.tokens2ids(sentence), 0) for sentence in
                             whole_data[whole_data["label"] == 0][:20000]["review"]]
        train_dataset = DataSet(self.train_data)
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)
        model = LSTM(len(self.vocab), self.embedding_dim, self.hidden_dim, self.num_class)
        model.to(self.device)
        nll_loss = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(self.num_epoch):
            total_loss = 0
            for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
                inputs, lengths, targets = [x.to(self.device) for x in batch]
                log_probs = model(inputs, lengths)
                loss = nll_loss(log_probs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Loss:{total_loss:.2f}")
        torch.save(model.state_dict(), self.model_file_name)
        with open(self.vocab_file_name, 'wb') as file:
            pickle.dump(self.vocab, file)

    def test_set(self):
        if not self.is_model_loaded :
            self.load_model()
            self.is_model_loaded = True
        whole_data = pd.read_csv(self.data_file_name, encoding='utf-8')
        self.test_data = [(self.vocab.tokens2ids(sentence), 1) for sentence in
                          whole_data[whole_data["label"] == 1][20000:]["review"]] \
                         + [(self.vocab.tokens2ids(sentence), 0) for sentence in
                            whole_data[whole_data["label"] == 0][20000:]["review"]]
        test_dataset = DataSet(self.test_data)
        test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=self.collate_fn, shuffle=False)
        acc = 0
        for batch in tqdm(test_data_loader, desc=f"Testing"):
            inputs, lengths, targets = [x.to(self.device) for x in batch]
            with torch.no_grad():
                output = self.model(inputs, lengths)
                acc += (output.argmax(dim=1) == targets).sum().item()
        print(f"ACC:{acc / len(test_data_loader):.2f}")
        return acc / len(test_data_loader)

    def predict_sentiment(self, text):
        if not self.is_model_loaded :
            self.load_model()
            self.is_model_loaded = True

        tokenized_text = jieba.lcut(text)
        input_ids = self.vocab.tokens2ids(tokenized_text)
        input_tensor = torch.tensor(input_ids).unsqueeze(0)  # 添加 batch 维度

        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            lengths = torch.tensor([len(input_ids)], dtype=torch.long)
            output = self.model(input_tensor, lengths)
            sentiment_prob = torch.exp(output).cpu().numpy()[0]  # 转回 CPU，并且从 tensor 提取出 numpy 数组
            positive_prob = sentiment_prob[1]  # 获取正面情感的概率

        return positive_prob

if __name__ == "__main__":
    print("test programa")
    print(torch.__version__)
    print(jieba.__version__)
    print(flask.__version__)
    saModel = SAModel()
    '''
    重新训练模型
    saModel.train()
    saModel.test_set()
    '''
    #saModel.train()
    text_to_predict = "速度很慢，格调也不足，油门又小"
    positive_prob = saModel.predict_sentiment(text_to_predict)
    print(f"Positive Probability: {positive_prob:.2f}")