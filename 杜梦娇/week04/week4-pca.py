import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None

    def fit_transform(self, X):

        # 1. 数据去中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. 计算协方差矩阵
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. 选择前n_components个主成分
        sorted_indices = np.argsort(eigenvalues)[::-1]  # 按特征值降序排序
        self.components_ = eigenvectors[:, sorted_indices[:self.n_components]]

        # 5. 将去中心化后的数据投影到主成分上
        X_pca = np.dot(X_centered, self.components_)

        return X_pca


# 测试代码
if __name__ == "__main__":
    # 生成随机数据
    X = np.array([[-1, 2, 66, -1],
                  [-2, 6, 58, -1],
                  [-3, 8, 45, -2],
                  [1, 9, 36, 1],
                  [2, 10, 62, 1],
                  [3, 5, 83, 2]])

    # 实现PCA降维到2维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print("原始数据形状:", X.shape)
    print("降维后的数据形状:", X_pca.shape)
    print("降维后的数据:\n", X_pca)
