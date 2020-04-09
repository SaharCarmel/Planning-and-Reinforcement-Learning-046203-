import numpy as np
import time


def Q2_DP_func(N, X):
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

                # dp_dict[x][n] = sum([k[n-1] for k in dp_dict[:x]])

    return sum([l[N - 1] for l in dp_dict])


s = time.time()
print(Q2_DP_func(12, 800))
print(f"Time: {time.time() - s}")
