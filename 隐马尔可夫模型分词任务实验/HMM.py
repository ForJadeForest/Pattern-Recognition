"""
使用已分词的文本进行训练
使用HMM解码算法对测试集进行分词
输出分词结果并写入test_result.txt
58119106 彭英哲 2021.5.30
"""
import numpy as np
import pandas as pd
import re
import itertools
import os
inf = -3.14e100  # 设置log（0）


def coding(string):
    """
    计算已分好词的字符串的隐状态序列
    :param string:已分好词的字符串
    :return:string对应的隐状态徐磊
    """
    obs = ''
    temp = string.split(' ')
    for word in temp:
        if word == '*':
            obs += word
            continue
        if len(word) == 1:
            obs += 'S'
        elif len(word) == 2:
            obs += 'BE'
        elif len(word) > 2:
            obs += 'B' + (len(word) - 2) * 'M' + 'E'
    return obs


def div_str(path, string, sep='/'):
    """
    依据path，对string进行分词
    :param sep: 字符串分割符
    :param path: string对应的隐状态序列
    :param string: 需要分词的字符串
    :return: 分词后的字符串
    """
    result = ''
    path = path[1:]  # 由于在decode函数求解path时没有用到path[0]来存储，所以要舍弃第一个元素
    for char, s in zip(path, string):
        result += s
        if char == 'S' or char == 'E':  # 只会在S，E后才可能分词结束
            result += sep
    return result


def cal_code_data(train_data):
    """
    使用*来对每一个句子划分，计算其隐状态序列
    :param train_data:
    :return: train_data中每一个句子对应的隐状态序列，划分的train_data
    """
    data = train_data.split('\n')
    data = [s for s in data]
    data = ' * '.join(data)
    return coding(data), data


class HMM:
    def __init__(self, train_data):
        """
        初始化HMM模型
        :param train_data: 文本训练集
        """
        self.index = ['B', 'M', 'E', 'S']
        A = self.cal_A(train_data)
        B, self.word_set = self.cal_B(train_data)
        S = self.cal_S(train_data)
        self.state_num = A.shape[0]
        self.view_num = B.shape[1]

        self.col = list(B.columns)
        self.A = pd.DataFrame(A, index=self.index, columns=self.index)
        self.B = B
        self.S = pd.Series(S, index=self.index)

    def cal_A(self, train_data):
        """
        计算状态转移矩阵
        :param train_data: 训练集
        :return: 状态转移矩阵
        """
        code_data = cal_code_data(train_data)[0]
        kind_list = list(itertools.product(*[self.index, self.index]))  # 生成四个隐状态的排列
        m = []
        for kind in kind_list:  # 计算每一种排列的出现的频率
            pattern = '{}(?={})'.format(kind[0], kind[1])
            pattern2 = '{}(?!,)'.format(kind[0])
            p = len(re.findall(pattern, code_data)) / len(re.findall(pattern2, code_data))
            m.append(p)

        A = np.array(m).reshape(4, 4)
        A = np.log(A)  # 对得到的状态转移矩阵取对数
        A[A == -np.inf] = inf  # 对log(0)取设定的负无穷
        return A

    def cal_B(self, train_data):
        """
        计算混淆矩阵
        :param train_data: 训练集
        :return: 混淆矩阵，训练集中的字符集合
        """
        code_data, data = cal_code_data(train_data)
        com_data = {
            'word': list(data.replace(' ', '')),
            'code_data': list(code_data)
        }
        df = pd.DataFrame(com_data)
        df = df[df['word'] != '*']  # 去掉设定的分隔符
        group_all = df.groupby(['word', 'code_data'])  # 对每个组合分组
        z = pd.DataFrame(group_all.size())
        word_set = np.unique([word[0] for word in z.index])
        B = pd.DataFrame(np.zeros((4, word_set.shape[0])), index=self.index, columns=word_set)
        for i in z.index:
            B.loc[i[1], i[0]] = z.loc[i, 0]
        new_B = B.T.apply(lambda x: x / x.sum()).T.apply(np.log)
        new_B[new_B == -np.inf] = inf
        return new_B, word_set

    def cal_S(self, train_data):
        """
        计算初始状态概率矩阵
        :param train_data: 训练集
        :return: 初始状态概率矩阵
        """
        S = pd.Series(np.zeros(4), index=self.index)
        code_data = cal_code_data(train_data)[0]
        title = pd.Series([s[0] for s in code_data.split('*')[:-1] if s != ''])
        title = pd.DataFrame(title.value_counts())
        for i in title.index:
            S[i] = title.loc[i]
        new_S = np.log(S / S.sum())
        new_S[new_S == -np.inf] = inf
        return new_S

    def decode(self, V):
        """
        定义：HMM的解码问题就是给定一个观测序列  ,找到最有可能（例如最能够解释观测序列，等等）的隐藏状态序列
        :param V: 已知的观测序列
        :return: S 最有可能的隐藏序列
        """
        time = len(V)
        view_col = list(range(1, time + 1))
        trellis = pd.DataFrame(np.zeros((self.state_num, time)), index=self.index, columns=view_col)
        l = pd.Series(V, index=view_col, name='Time')
        path = pd.DataFrame(np.zeros((self.state_num, time)), index=self.index, columns=view_col)

        trellis.loc[:, 1] = self.S + self.B.loc[:, self.word_set[l[1]]]  # trellis 的初始化
        for t in range(2, time + 1):
            for j in self.index:
                temp = trellis.loc[:, t - 1] + self.A.loc[:, j]
                # t-1时刻的状态i的trellis 乘状态转移参数 a_{i,j}转移致j状态的概率
                # 得到隐状态的个数的概率结果，我们只要取最大的即可
                trellis.loc[j, t] = temp.max() + self.B.at[j, self.word_set[l[t]]]
                # 计算转移至j状态发出t时刻可见状态的概率
                path.loc[j, t] = temp.idxmax()
                # 保存此时的状态名
        p_time = pd.Series(np.zeros(time))
        p_time[time] = trellis.loc[:, time].idxmax()
        for i in range(1, time):
            p_time[time - i] = path.loc[p_time[time - i + 1], time - i + 1]
        return p_time

    def get_index(self, string, word_set):
        """
        返回每个字符对应的索引序列
        :param string: 预测用的字符串
        :param word_set: 训练集的字符集
        :return: 字符串对应的字符序列
        """
        index = []
        for i in string:
            if i in word_set:
                index.append(np.where(word_set == i)[0][0])
            else:  # 如果不存在字符集，则加入到字符集中，并指定为M
                self.B.loc['M', i] = 0
                self.word_set = np.append(self.word_set, i)
                index.append(self.B.shape[1] - 1)
                self.view_num = self.B.shape[1]
                self.col = list(self.B.columns)
        return index


current_path = os.getcwd()
train_path = current_path + os.sep + 'data' + os.sep + 'RenMinData.txt_utf8'
with open(train_path, encoding='utf8')as f:
    text = f.read()
h = HMM(text)

str_list = [
    '今天我来到了东南大学。',
    '模式识别课程是一门有趣的课程。',
    '我认为完成本次实验是一个挑战。',
    '我今天花了整整3小时辛辛苦苦终于写出来了隐马尔可夫模型的代码！',
    '人生犹如花草，容颜终将在岁月的风尘中老去。能在时光的雕刻刀下，让自己保留清晨阳光般的笑容，端庄厚重的气度，深刻内敛的内涵，那将是上苍赐予我们一生最宝贵的财富。',
    '烟水两茫茫，蒹葭复苍苍，你就是那一位伫立于水之湄的俏佳人，着一袭素裳霓裙，黛眉如远山，一双含情眸如一池碧波，秋水盈盈，潋滟妩媚，清雅逼人，皓肤若凝脂，冰肌似玉骨！',
    '我们生活在一个安定团结的国家中。'
]
for s in str_list:
    res = div_str(h.decode(h.get_index(s, h.word_set)), s)
    with open('test_result.txt','a')as f:
        f.write('原始句子：'+'\n'+s+'\n'+'分词后句子：'+'\n'+res+'\n')
        f.write('\n')

