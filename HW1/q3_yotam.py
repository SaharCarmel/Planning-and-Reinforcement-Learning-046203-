import numpy as np
import time


def q2_dp_func(N, X):
    dp_dict = np.zeros([X + 1, N + 1])
    for x in range(1, X):
        for n in range(1, N + 1):
            if n > x:
                dp_dict[x][n] = 0
            elif n == x or n == 1:
                dp_dict[x][n] = 1
            else:
                for k in range(x):
                    dp_dict[x][n] += dp_dict[k][n - 1]
    return sum([l[N - 1] for l in dp_dict])


s = time.time()
print(q2_dp_func(12, 800))
print(f"Time: {time.time() - s}")
