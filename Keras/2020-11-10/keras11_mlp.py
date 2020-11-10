
# 1. 데이터

import numpy as np

x = np.array([range(1, 101), range(311, 411), range(100)])
x = np.transpose(x) # 전치 행렬

y = np.array([range(101, 201), range(711, 811), range(100)])
y = np.transpose(y)


print(x.shape) # (3, 1) 스칼라가 3개
print(y.shape) # (3, 100)

# (100, 3)

#스칼라 / 벡터 / 행렬 / 텐서
# https://art28.github.io/blog/linear-algebra-1/
# 스칼라 : 하나의 숫자
# 벡터 : 숫자의 배열
# 행렬 : 2차원 배열
# 텐서 : 2차원 이상의 배열



