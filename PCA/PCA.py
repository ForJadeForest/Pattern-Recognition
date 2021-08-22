import numpy as np
import matplotlib.pyplot as plt


def pca(train_data, target_dimension):
    """
    calculate the S
    :param target_dimension: the target reduction dimension of the train data
    :param train_data: the train_data (type: ndarray n*p)
    :return:1) the result of data reduced and the transformation matrix
    """
    standard_data = train_data - train_data.mean(axis=0)
    # 中心化
    scatter_matrix = standard_data.T.dot(standard_data)
    print(scatter_matrix)
    # calculate the scatter matrix
    eigenvalue, eigenvector = np.linalg.eig(scatter_matrix)
    print(eigenvalue,eigenvector)
    # calculate the scatter matrix's eigenvector and eigenvalue
    rel = list(zip(abs(eigenvalue), eigenvector.T))
    rel.sort(key=lambda eig: eig[0], reverse=True)
    # sort the eigenvalue
    trans_matrix = np.concatenate([rel[i][1] for i in range(target_dimension)]).reshape((-1, target_dimension),
                                                                                        order='F')
    # concat the eigenvector
    return trans_matrix.T.dot(standard_data.T), trans_matrix


