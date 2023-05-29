import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from matplotlib import pyplot as plt
import numpy as np
import random
from IPython import display

#############################################
#随机
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
#顺序
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader: 
    """加载序列数据的迭代器"""
    def __init__(self, x,batch_size, num_steps, use_random_iter):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus = x
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_x_data(x,batch_size, num_steps, 
                           use_random_iter=False):
    data_iter = SeqDataLoader(
        x,batch_size, num_steps, use_random_iter)
    return data_iter
###########################################################







class RNNModelScratch:
    def __init__(self,vocab,num_hiddens,get_params,init_state,forward_fn):
        self.vocab,self.num_hiddens = vocab,num_hiddens
        self.params = get_params(vocab,num_hiddens)
        self.init_state,self.forward_fn = init_state,forward_fn 
    
    def __call__(self,X,state):
        #原先X大小：batch_size x num_steps
        #现在：num_steps x batch_size x 1
        X = X.T
        X = X.view(X.shape[0],X.shape[1],1)
        return self.forward_fn(X,state,self.params)
    
    def begin_state(self,batch_size):
        return self.init_state(batch_size,self.num_hiddens)

def predict(prefix,num_preds,net,vocab):
    state = net.begin_state(batch_size=1)
    outputs = [prefix[0]]
    get_input = lambda: torch.tensor([outputs[-1]]).reshape((1,1))
    for y in prefix[1:]:
        _,state = net(get_input(),state)
        outputs.append(y)
    for _ in range(num_preds):
        y,state = net(get_input(),state)
        outputs.append(y)
    return outputs

def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch(net,train_iter,loss,updater,use_random_iter):
    state = None
    for X,Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0])
        else:
            if isinstance(net,nn.Module) and not isinstance(state,tuple):
                #GRU
                state.detach_()
            else:
                for s in state:
                    s.detach_()
    # y = Y.reshape(-1)
    y_hat,state = net(X,state)
    l = loss(y_hat,Y.reshape(y_hat.shape)).sum()
    if isinstance(updater,torch.optim.Optimizer):
        updater.zero_grad()
        l.backward()
        grad_clipping(net,1)
        updater.step()
    else:
        l.backward()
        grad_clipping(net,1)
        updater(batch_size=1)
    return l.sum()

def train(net,train_iter,vocab,lr,num_epochs,use_random_iter=False):
    animator = Animator(xlabel='epoch',ylabel='loss', xlim=[1, num_epochs], ylim=[0, 0.9])
    loss = nn.MSELoss()
    plot_l = []
    if isinstance(net,nn.Module):
        updater = torch.optim.Adam(net.parameters(),lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    # predict = lambda prefix: predict(prefix, 5, net, vocab)
    # ppl=[]
    for epoch in range(num_epochs):
        ppl = train_epoch(
            net, train_iter, loss, updater, use_random_iter)
        # plot_l.append(ppl)
        if (epoch + 1) % 20 == 0:
            # print(predict(torch.tensor([1.0,2.0,3.0],dtype=torch.float32)))
            # print(f"Iter: {epoch+1} loss: {ppl} ")
            # print(ppl)
            # print(ppl)
            l = ppl.tolist()
            animator.add(epoch+1,l)





###########################################################
#简洁实现
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        #原先inputs大小：batch_size x num_steps
        #现在：num_steps x batch_size x 1
        X = inputs.T
        X = X.view(X.shape[0],X.shape[1],1)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        #(128,1)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens)),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens)))
################################################################################################
##########################################Animator######################################################
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
################################################################################################