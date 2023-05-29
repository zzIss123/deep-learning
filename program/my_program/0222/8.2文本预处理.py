import numpy as np
import collections
import re 
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt','090b5e7e70c295757f55df93cb0a180b9691891a')

lines = d2l.read_time_machine()
# print(f'# 文本总行数: {len(lines)}')
# print(f'shape: {np.shape(lines)}')
# print(lines[0])
# print(lines[10])
# print(lines[100])
tokens = d2l.tokenize(lines)
for i in range(11):
    print(tokens[i])
vocab = d2l.Vocab(tokens)
# print(list(vocab.token_to_idx.items())[:10])
# 
# print('--------------------------------------')
# 
# for i in [0,10]:
#     print('文本',tokens[i])
#     print('索引',vocab[tokens[i]])

corpus, vocab = d2l.load_corpus_time_machine()
# print(len(corpus))
# print(len(vocab))
