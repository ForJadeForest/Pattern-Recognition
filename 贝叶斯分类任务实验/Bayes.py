"""
使用wine葡萄酒数据集进行贝叶斯分类
并对测试集进行预测
输出预测结果以及相关的概率写入test_prediction.csv
58119106 彭英哲 2021.5.31
"""

import pandas as pd
import numpy as np
import os

current_path = os.getcwd()
train_path = current_path + os.sep + 'data/train_data.csv'
test_path = current_path + os.sep + 'data/test_data.csv'


class Bayes:
    def __init__(self, train_x, train_y):
        """
        初始化模型
        :param train_x:训练集的特征，pandas.DataFrame格式
        :param train_y:训练集的标签，pandas.Series格式
        """
        self.train_x = train_x
        self.train_y = train_y
        self.pdf = {}  # 计算每一个类别不同特征的条件概率密度
        self.labels = train_y.unique()
        for label in self.labels:
            index = train_y[train_y == label].index
            self.pdf[label] = {
                'mean': train_x.loc[index].mean(),
                'std': train_x.loc[index].std()
            }
        self.pri = self.cal_pri(self.labels)  # 计算不同类别的先验概率

    def cal_pri(self, data_label):
        """
        计算每个类别的先验概率
        :param data_label:训练集所有标签
        :return: 所有标签的先验概率
        """
        label_num = []
        for attr_label in self.labels:
            label_num.append(data_label[data_label == attr_label].shape[0])
        return pd.DataFrame(label_num, index=self.labels)  # 将标签设置为index

    def predict(self, data):
        """
        预测未知数据
        :param data:需要预测的样本矩阵，pandas.DataFrame格式
        :return: 1)预测对应的标签
                 2）分属于3个类别的概率
        """
        pred = []
        all_prob = []
        for sample in data.index:
            prob = pd.Series(np.zeros_like(self.labels), index=self.labels)
            x = data.loc[sample]
            for label in self.labels:
                total_p = 1
                for attr in self.train_x.columns:
                    total_p *= Gaussian(mu=self.pdf[label]['mean'][attr], gamma=self.pdf[label]['std'][attr], x=x[attr])
                    # 乘以每一个特征对于attr（label）的pdf
                prob.loc[label] = total_p * self.pri.loc[label, 0]
                # 最后再乘以先验概率
            pred.append(prob.idxmax())
            all_prob.append(prob)
        return pred, all_prob

    def score(self, x, y):
        """
        计算预测的正确率
        :param x: 预测所用的样本
        :param y: 预测数据对于标签
        :return: 准确率
        """
        pred = self.predict(x)[0]
        res = [i for i in range(len(pred)) if pred[i] == y[i]]
        return len(res) / len(pred)


def Gaussian(mu, gamma, x):
    """
    计算1维高斯分布的概率
    :param mu: 均值
    :param gamma: 方差
    :param x: 输入样本的特征
    :return: 高斯分布的对应概率
    """
    return 1 / np.sqrt(2 * np.pi * gamma ** 2) * np.exp(-np.square(x - mu) / 2 / np.square(gamma))


def load_data(data_path):
    """
    加载数据，并分成标签和特征矩阵
    ！！数据的标签在第一列，后面13全为特征！！
    :param data_path: 数据对应地址,str
    :return: 返回整个数据矩阵，特征矩阵，所有标签
    """
    df = pd.read_csv(data_path, header=None, names=['label'] + [i for i in range(13)])
    x = df.iloc[:, 1:]
    y = df.loc[:, 'label']
    return df, x, y


def cross_validation(k, data):
    """
    进行交叉验证
    :param model:训练的模型
    :param k: 折数
    :param data: 训练数据 pandas格式
    :return: 正确率列表
    """
    test_length = data.shape[0] // k
    data = data.sample(frac=1).reset_index(drop=True)
    data_part = [
        data.iloc[i * test_length:(i + 1) * test_length] if (i + 1) != k else data.iloc[i * test_length:] for i in
        range(k)
    ]
    acc = []
    for i in range(k):
        x_test = data_part[i].iloc[:, 1:].reset_index(drop=True)
        y_test = data_part[i].iloc[:, 0].reset_index(drop=True)
        train_data = pd.concat([d for num, d in enumerate(data_part) if num != i])
        x_train = train_data.iloc[:, 1:]
        y_train = train_data.loc[:, 'label']
        model = Bayes(x_train, y_train)

        acc.append(model.score(x_test, y_test))
    return acc


data, train_x, train_y = load_data(train_path)
model = Bayes(train_x, train_y)
test_x, test_y = load_data(test_path)[1:]
print('测试集准确率', model.score(test_x, test_y))

pred, prob = model.predict(test_x)
pred = pd.DataFrame(pred)
prob = pd.DataFrame(prob).apply(lambda x: x / x.sum(), axis=1).apply(lambda x: round(x, 3))
cat = pd.concat([pred, prob], axis=1)
cat.to_csv('test_prediction.csv', header=False, index=False)
print('文件保存完成')
print('开始交叉验证检验模型 k = 5')
acc = [round(i,3) for i in cross_validation(5, data)]
print('正确率列表：')
print(acc)
print('正确率均值：')
print(sum(acc)/len(acc))

print('开始交叉验证检验模型 k = 10')
acc = [round(i,3) for i in cross_validation(20, data)]
print('正确率列表：')
print(acc)
print('正确率均值：')
print(sum(acc)/len(acc))
