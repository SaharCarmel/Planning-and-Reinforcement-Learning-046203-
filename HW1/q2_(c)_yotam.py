import numpy as np
import time


def q2_dp_func(N, X):
    dp_mat = np.zeros([N + 1, X + 1])
    for i in range(1, N+1):
        for j in range(1, X + 1):
            if i > j:
                dp_mat[i][j] = 0
            elif i == j or i == 1:
                dp_mat[i][j] = 1
            else:
                dp_mat[i][j] = dp_mat[i-1][j - 1] + dp_mat[i][j - 1]
    return dp_mat[N][X]


s = time.time()
print(q2_dp_func(12, 800))
print(f"Time: {time.time() - s}")
