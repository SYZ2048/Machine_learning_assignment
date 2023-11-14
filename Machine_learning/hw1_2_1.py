import numpy as np


def topk(mat, k):
    e_vals, e_vecs = np.linalg.eig(mat)
    sorted_indices = np.argsort(e_vals)
    return e_vals[sorted_indices[:-k - 1:-1]], e_vecs[:, sorted_indices[:-k - 1:-1]]


X1 = np.array([[2, 3, 3, 4, 5, 7], [2, 4, 5, 5, 6, 8]])
X2 = np.array([[-1, -1, 0, 2, 0], [-2, 0, 0, 1, 1]])

X1 = X1 - np.average(X1, axis=1).reshape(2, 1)
X2 = X2 - np.average(X2, axis=1).reshape(2, 1)
print(X1)

e_vals1, e_vecs1 = topk(np.dot(X1, X1.T) / len(X1[0]), 1)
e_vals2, e_vecs2 = topk(np.dot(X2, X2.T) / len(X2[0]), 1)
print(e_vals1)
print(e_vecs1)

print(np.dot(e_vecs1.T, X1))
print(np.dot(e_vecs2.T, X2))

K = np.zeros((len(X2[0]), len(X2[0])))

for i in range(len(X2[0])):
    for j in range(len(X2[0])):
        K[i][j] = np.dot(X2[:, i], X2[:, j]) + 1

K = K - np.average(K, axis=1).reshape(5, 1)
e_vals3, e_vecs3 = topk(K, 1)

print(np.dot(e_vecs3.T, K) / e_vals3 ** (1 / 2))
