import numpy as np


def LDA(data, labels):
    class_label = np.unique(labels)
    S_w = np.zeros(shape=(data.shape[1], data.shape[1]))
    for label in class_label:
        temp = data[labels == label]
        standard = temp - temp.mean(axis=0)
        s_w = standard.T.dot(standard)
        S_w += s_w
    aver_all = data.mean(axis=0)
    S_b = np.zeros(shape=(data.shape[1], data.shape[1]))
    for label in class_label:
        temp = data[labels == label]
        s_b = temp.shape[0] * (temp.mean(axis=0) - aver_all).T.dot(temp.mean(axis=0) - aver_all)
        S_b += s_b
    eigenvalue, eigenvector = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))
    rel = list(zip(abs(eigenvalue), eigenvector.T))
    rel.sort(key=lambda eig: eig[0], reverse=True)
    trans_matrix = np.concatenate([rel[i][1] for i in range(1)]).reshape((-1, 1), order='F')
    return trans_matrix.T.dot(data.T), trans_matrix
