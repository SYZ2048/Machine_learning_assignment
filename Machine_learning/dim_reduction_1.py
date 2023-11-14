import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE

# 生成三维螺旋线数据
t = np.linspace(0, 4 * np.pi, 1000)
R = 3
x = R * np.cos(t)
y = R * np.sin(t)
z = t
X = np.column_stack([x, y, z])  # (1000, 3)
print(X.shape)

fig = plt.figure(figsize=(24, 6))

# 绘制原始螺旋线数据
ax1 = fig.add_subplot(1, 6, 1, projection='3d')
ax1.scatter(x, y, z, c=t)
ax1.set_title('Raw Spiral Data')

# PCA
X_pca = PCA(n_components=2).fit_transform(X)    # Fit the model from data in X and transform X.
ax2 = fig.add_subplot(1, 6, 2)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=t)
ax2.set_title('PCA')

# MDS
X_mds = MDS(n_components=2).fit_transform(X)
ax3 = fig.add_subplot(1, 6, 3)
ax3.scatter(X_mds[:, 0], X_mds[:, 1], c=t)
ax3.set_title('MDS')

# IsoMap
X_isomap = Isomap(n_components=2).fit_transform(X)
ax4 = fig.add_subplot(1, 6, 4)
ax4.scatter(X_isomap[:, 0], X_isomap[:, 1], c=t)
ax4.set_title('IsoMap')

# LLE
X_lle = LocallyLinearEmbedding(n_components=2).fit_transform(X)
ax5 = fig.add_subplot(1, 6, 5)
ax5.scatter(X_lle[:, 0], X_lle[:, 1], c=t)
ax5.set_title('LLE')

# t-SNE
X_tsne = TSNE(n_components=2).fit_transform(X)
ax6 = fig.add_subplot(1, 6, 6)
ax6.scatter(X_tsne[:, 0], X_tsne[:, 1], c=t)
ax6.set_title('t-SNE')

plt.tight_layout()
plt.show()
