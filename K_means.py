from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import numpy as np

np.random.seed(42)

# 这里使用load_digits函数从Scikit-learn库中加载手写数字数据集。X_digits包含图像数据，而y_digits包含相应的标签（0到9）
X_digits, y_digits = load_digits(return_X_y=True)  # len(y_digits) = 1797
# train_test_split用于将数据分为训练集和测试集，设定random_state保证结果可重复，以默认的0.25作为分割比例进行分割（训练集:测试集=3:1）
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)
# print(len(y_train))   1347
# print(len(y_test))    450
n_labeled = 50  # 意味着只有50个样本是有标签的，将被用于训练逻辑回归模型

# Baseline
# LogisticRegression: 这是一个逻辑回归模型，用于分类任务     multi_class : 告知模型要处理的分类问题是二分类还是多分类，ovr（二分类）
# solver="lbfgs"拟牛顿法的一种利用
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", random_state=42)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])  # 拟合
baseline = log_reg.score(X_test, y_test)
# 定义并训练一个逻辑回归模型使用前50个样本。然后使用测试集评估其性能。
print('Baseline Accuracy: %.2f' % (baseline * 100))

# You should work here
# k-means聚类，找到k个代表性的样本
k = 100
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)  # Compute clustering and transform X to cluster-distance space.
# np.argmin给出axis方向最小值的下标,比较j取同一值时，不同的i所指向的元素之间的大小关系
representative_digit_idx = np.argmin(X_digits_dist, axis=0) # 寻找离集群中心最近的点做样本点
X_representative_digits = X_train[representative_digit_idx]
y_representative_digits = y_train[representative_digit_idx]

# 训练逻辑回归模型
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
result = log_reg.score(X_test, y_test)

print("After K-means Accuracy: %.2f" % (result * 100))
