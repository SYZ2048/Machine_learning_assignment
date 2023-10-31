import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE

# 生成双螺旋结构的数据
t = np.linspace(0, 4 * np.pi, 1000)  # 生成t值
R = 3
x1 = R * np.sin(t)
y1 = R * np.cos(t)
z1 = t

x2 = R * np.sin(t + np.pi)  # 相位差pi
y2 = R * np.cos(t + np.pi)
z2 = t

# 合并数据
X1 = np.column_stack([x1, y1, z1])
X2 = np.column_stack([x2, y2, z2])


# 创建3D螺旋线可视化的函数
def plot_raw_3d_data(x1, y1, z1, x2, y2, z2, ax):
    ax.scatter(x1, y1, z1, c=t)
    ax.scatter(x2, y2, z2, c=t)
    ax.set_title('Raw Spiral Data')
    ax.view_init(elev=20, azim=60)  # 调整视角


# 将数据降至二维并进行可视化的函数
def reduce_dim_and_plot(X, t_values, method, title, ax):
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

    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t_values)
    ax.set_title(title)


methods = ['PCA', 'MDS', 'Isomap', 'LLE', 't-SNE']

# 将两条螺旋线当作两个分布，分别降维
fig = plt.figure(figsize=(45, 25))
ax_3d = fig.add_subplot(len(methods), 3, 1, projection='3d')    # 绘制原始三维螺旋线
plot_raw_3d_data(x1, y1, z1, x2, y2, z2, ax_3d)
for i, method in enumerate(methods):
    ax_2d_1 = fig.add_subplot(len(methods), 3, i*3+2)
    reduce_dim_and_plot(X1, t, method, f'{method} - Spiral 1', ax_2d_1)
    ax_2d_2 = fig.add_subplot(len(methods), 3, i*3+3)
    reduce_dim_and_plot(X2, t, method, f'{method} - Spiral 2', ax_2d_2)

plt.tight_layout()
plt.show()

# 将两条螺旋线当作一个分布，一起降维
X_combined = np.vstack([X1, X2])
t_combined = np.concatenate([t, t])

fig = plt.figure(figsize=(36, 15))
ax_3d = fig.add_subplot(len(methods), 2, 1, projection='3d')
plot_raw_3d_data(x1, y1, z1, x2, y2, z2, ax_3d)
for i, method in enumerate(methods):
    ax_2d_combined = fig.add_subplot(len(methods), 2, i*2+2)
    reduce_dim_and_plot(X_combined, t_combined, method, f'{method} - Combined', ax_2d_combined)

plt.tight_layout()
plt.show()
