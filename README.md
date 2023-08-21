#### 中文文本情感分析模型（以汽车评价为例）项目文档

​	作者：张洺玮

​	学号：202234071006

​	邮箱：13337649640@163.com

##### 1.介绍

​	文本情感分析在商业评估预测、舆情分析方面都有十分广阔的应用前景。本项目作为课程的作业项目，以python为编程语言，PyTorch为深度学习框架，构建了 LSTM 模型用于中文文本的情感分析。此外，在demo中利用了 Flask 框架创建了一个用户友好的 Web 用户界面，可实现输入文本情感分析以及测试集测试功能。由于数据集及本机算例的缘故，仅针对汽车评价问题，使用汽车评价数据集进行训练和预测。

​	本项目的优点是选题富有意义、代码可读性强、各类数据及模型均可直接加载预测、项目文件管理井井有条；不足是数据相对较少，仅适用汽车评价进行训练，普适性较差。

​	数据集由汽车方面的35000条正向评论以及35000条负面评论组成，来源：https://download.csdn.net/download/weixin_42756970/87352958

##### 2.依赖包版本

​	Python == 3.9.7

​	PyTorch == 2.0.1+cu117

​	jieba == 0.42.1

​	flask ==  1.1.2

##### 3.主要脚本功能

​	main.py - 构建用户界面

​	sentiAnalysisModel.py - 核心代码，构建模型，封装接口

##### 4.如何使用

###### 	**4.1运行demo**

​		`./demo.bat`或点击demo.bat文件即可进入ui界面，程序执行过程可在终端查看，用于分析的文本可直接从test_sentence.txt中复制。请注意，模型已经训练至一个较好的状态，重新训练后可能效果会降低。输入文本请尽量与汽车的体验有关。

###### 	**4.2接口调用**

​		引入接口

```python
from sentiAnalysisModel import *
from sentiAnalysisModel import SAModel
```

​		建立模型对象

```python
model = SAModel()
```

​		训练模型

```python
model.train()
```

​		测试测试集

```
model.test_set()
```

​		分析文本

```python
positive_prob = model.predict_sentiment(text_to_predict)
```

##### 5.常见错误

​	如果ui界面提示服务器忙碌，可能由程序初始化未完成导致，稍等刷新界面即可。其余错误一般由依赖包版本不匹配导致，可`python  sentiAnalysisModel.py`根据终端报错排查。

##### 6.demo演示

###### 	6.1ui界面

<img src=".\img\image1.png" alt="s" style="zoom:67%;" />

###### 	6.2准确率

​	使用40000条数据进行训练，30000条数据进行评估，准确率为94%

<img src=".\img\image2.png" alt="image2" style="zoom: 67%;" />

##### 7.核心代码分析

###### 	7.1数据处理

​	封装Vocab类构建词汇表以及将文本数据映射为数字索引，并使用停用词表对数据进行过滤。

```python
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
```

###### 	7.2网络结构

​	封装LSTM类实现LSTM模型的网络结构。网络分为词向量层，将输入的token索引映射为embedding向量；LSTM层，即长短时记忆网络层，用于学习序列信息； 输出层，线性变换输出分类结果。

```python
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
```

###### 	7.3模型封装

​	将模型进行封装，统一管理文件地址、超参数等，将训练、加载、评估、预测等功能进行封装。

```python
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
...
```

##### 8.参考资料

​	**论文**

[1] Paszke, Adam , et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." (2019).

[2] Kim, Yoon . "Convolutional Neural Networks for Sentence Classification." Eprint Arxiv (2014).

​	**博客**

[1]Pytorch实现文本情感分析:https://blog.csdn.net/tcn760/article/details/125036931

[2]自然语言处理实战——Pytorch实现基于LSTM的情感分析(LMDB)——详细:https://blog.csdn.net/m0_53328738/article/details/128367345

[3]情感分析之电影评论分析-基于Tensorflow的LSTM:https://blog.csdn.net/lilong117194/article/details/82217271
