from random import random
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE

# choose motion (corresponding to question number)
motion = 2  # 1 or 2


num_points = 500
cmap1 = plt.cm.hot
cmap2 = plt.cm.winter
methods = ['PCA', 'MDS', 'Isomap', 'LLE', 't-SNE']


# 将数据降至二维并进行可视化的函数
def reduce_dim_and_plot(X, z_values, method, title, ax):
    if method == 'PCA':
        X_reduced = PCA(n_components=2).fit_transform(X)
    elif method == 'MDS':
        X_reduced = MDS(n_components=2).fit_transform(X)
    elif method == 'Isomap':
        X_reduced = Isomap(n_components=2).fit_transform(X)
    elif method == 'LLE':
        X_reduced = LocallyLinearEmbedding(n_components=2).fit_transform(X)
    elif method == 't-SNE':
        X_reduced = TSNE(n_components=2).fit_transform(X)
    else:
        raise ValueError('Unknown method')

    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=z_values, cmap=cmap2)
    ax.set_title(title)


def single_spiral_dim_reduction():   # version 1
    z = np.linspace(0, 4 * math.pi, num_points)  # (500,)
    z = z.reshape(1, -1)  # (1, 500)
    x = 2 * np.cos(z) + np.random.rand(num_points)
    y = 2 * np.sin(z) + np.random.rand(num_points)

    data = np.concatenate((x, y, z))  # (3, 500)
    color = np.linspace(0, 1, x.shape[1])

    # Work Here:syzedit
    X = data.T

    # draw raw spiral data
    fig = plt.figure(figsize=(30, 6))
    ax1 = fig.add_subplot(1, 6, 1, projection='3d')
    ax1.scatter(x, y, z, c=color, cmap=cmap1)
    ax1.set_title('Raw Spiral Data')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    for i, method in enumerate(methods):
        ax = fig.add_subplot(1, len(methods)+1, i+2)
        reduce_dim_and_plot(X, z, method, f'{method}', ax)

    plt.tight_layout()
    plt.show()


def dual_spiral_dim_reduction():
    z = np.linspace(0, 4 * math.pi, num_points)
    z = z.reshape(1, -1)
    x1 = 2 * np.cos(z) + np.random.rand(num_points)
    y1 = 2 * np.sin(z) + np.random.rand(num_points)
    x2 = 2 * np.sin(z) + np.random.rand(num_points)
    y2 = 2 * np.cos(z) + np.random.rand(num_points)

    data1 = np.concatenate((x1, y1, z))
    data2 = np.concatenate((x2, y2, z))
    color1 = np.linspace(0, 1, x1.shape[1])
    color2 = np.linspace(0, 1, x2.shape[1])
    X1 = data1.T
    X2 = data2.T

    # 将两条螺旋线当作两个分布，分别降维
    fig = plt.figure(figsize=(45, 25))
    ax1 = fig.add_subplot(len(methods), 3, 1, projection='3d')  # 绘制原始三维螺旋线
    # plot_raw_3d_data(x1, y1, z, x2, y2, z, ax1, color1, color2, cmap1, cmap2)
    ax1.scatter(x1, y1, z, c=color1, cmap=cmap1)
    ax1.scatter(x2, y2, z, c=color2, cmap=cmap2)
    # ax1.view_init(elev=20, azim=60)  # 调整视角
    ax1.set_title('Raw Spiral Data')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    for i, method in enumerate(methods):
        ax_2d_1 = fig.add_subplot(len(methods), 3, i * 3 + 2)
        reduce_dim_and_plot(X1, z, method, f'{method} - Spiral 1', ax_2d_1)
        ax_2d_2 = fig.add_subplot(len(methods), 3, i * 3 + 3)
        reduce_dim_and_plot(X2, z, method, f'{method} - Spiral 2', ax_2d_2)

    # 将两条螺旋线当作一个分布，一起降维
    X_combined = np.vstack([X1, X2])
    z_combined = np.concatenate([z, z])
    fig = plt.figure(figsize=(10, 15))
    # 绘制原始三维螺旋线
    ax1 = fig.add_subplot(len(methods), 2, 1, projection='3d')
    ax1.scatter(x1, y1, z, c=color1, cmap=cmap1)
    ax1.scatter(x2, y2, z, c=color2, cmap=cmap2)
    # ax1.view_init(elev=20, azim=60)  # 调整视角
    ax1.set_title('Raw Spiral Data')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    for i, method in enumerate(methods):
        ax_2d_combined = fig.add_subplot(len(methods), 2, i * 2 + 2)
        reduce_dim_and_plot(X_combined, z_combined, method, f'{method} - Combined', ax_2d_combined)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if motion == 1:
        single_spiral_dim_reduction()
    elif motion == 2:
        dual_spiral_dim_reduction()
