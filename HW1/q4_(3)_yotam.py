import numpy as np


def LCS_3_Length(X, Y, Z):
    m = len(X)
    n = len(Y)
    l = len(Z)

    c = np.zeros([m + 1, n + 1, l + 1])
    b = np.zeros([m + 1, n + 1, l + 1]).tolist()

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            for k in range(1, l + 1):
                if X[i - 1] == Y[j - 1] == Z[k - 1]:
                    c[i][j][k] = c[i - 1][j - 1][k - 1] + 1
                    b[i][j][k] = "up-left-in"
                elif c[i - 1][j][k] >= c[i][j - 1][k] and c[i - 1][j][k] >= c[i][j][k - 1]:
                    c[i][j][k] = c[i - 1][j][k]
                    b[i][j][k] = "up"
                elif c[i][j-1][k] >= c[i-1][j][k] and c[i][j-1][k] >= c[i][j][k - 1]:
                    c[i][j][k] = c[i][j-1][k]
                    b[i][j][k] = "left"
                else:
                    c[i][j][k] = c[i][j][k-1]
                    b[i][j][k] = "in"
    return b, c


LCS_3_Length("ABCA", "ACBC", "BABC")

#
# def LCS_2_Length(X, Y):
#     m = len(X)
#     n = len(Y)
#
#     c = np.zeros([m + 1, n + 1])
#     b = np.zeros([m + 1, n + 1]).tolist()
#
#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             if X[i - 1] == Y[j - 1]:
#                 c[i][j] = c[i - 1][j - 1] + 1
#                 b[i][j] = "nw-arrow"
#             elif c[i - 1][j] >= c[i][j - 1]:
#                 c[i][j] = c[i - 1][j]
#                 b[i][j] = "up-arrow"
#             else:
#                 c[i][j] = c[i][j - 1]
#                 b[i][j] = "left-arrow"
#     return b, c
#
#
# LCS_2_Length("ABCA", "ACBC")
