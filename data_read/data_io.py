# coding=utf-8
"""
按字符表示数据处理
"""

data = open('data.txt', 'r').read() # 读入数据
chars = list(set(data)) # 每个字符的list(字符词典)
vocab_size = len(chars)
data_size = len(data)
char_to_id = {ch:i for i,ch in enumerate(chars)} # enumerate在字典上是枚举、列举的意思 (0, seq[0]), (1, seq[1]), (2, seq[2])
id_to_char = {i:ch for i,ch in enumerate(chars)}