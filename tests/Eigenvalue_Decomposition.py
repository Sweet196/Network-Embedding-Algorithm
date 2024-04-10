import torch

# 输入一个3x3矩阵
A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])

# 使用torch.eig进行特征值分解
eigenvalues, eigenvectors = torch.linalg.eig(A)

# 输出特征值和特征向量
print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)
