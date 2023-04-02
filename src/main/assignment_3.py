import numpy as np
import sys
def dy(t, y):
    return t - y ** 2

# Question 1: Euler Method with the following details
# Function: t - y^2, Range: 0 < t < 2
# Iterations: 10, Initial Point f(0) = 1
def euler(f, low, high, it, x0, y0):
    h = (high - low) / it
    t = x0
    w = y0

    for i in range(it):
        w += h * f(t, w)
        t += h

    return w

# Question 2: Runge-Kutta with the following details
# Function: t - y^2, Range 0 < t < 2
# Iterations: 10, Initial Point f(0) = 1
def runge_kutta(f, low, high, it, x0, y0):
    h = (high - low) / it
    t = x0
    w = y0
    for i in range(it):
        k1 = h * f(t, w)
        k2 = h * f(t + h / 2, w + k1 / 2)
        k3 = h * f(t + h / 2, w + k2 / 2)
        k4 = h * f(t + h, w + k3)

        w = w + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h

    return w

# Question 3: Use Gaussian elimination and backward
# substitution solve the following linear sytem of
# equations written in augmented matrix format.
# [  2 -1  1 |  6]
# |  1  3  1 |  0|
# [ -1  5  4 | -3]
def gaussian(matrix):
    n = len(matrix)
    for i in range(n - 1):
        p = -1
        for j in range(i, n):
              if matrix[j][i] != 0:
                  p = j
                  break
        if p == -1:
            return -1
        if (p != i):
            temp = matrix[p].copy()
            matrix[p] = matrix[i]
            matrix[i] = temp

        for j in range(i + 1, n):
            m = matrix[j][i] / matrix[i][i]
            for k in range(len(matrix[i])):
                matrix[j][k] -= m * matrix[i][k]
        
        if matrix[n - 1][n - 1] == 0:
            return -1

        x = np.zeros(n)
        x[n - 1] = matrix[n - 1][n] / matrix[n - 1][n - 1]

        for i in range(n - 2, -1, -1):
            x[i] = matrix[i][n]
            for j in range(i + 1, n):
                x[i] -= matrix[i][j] * x[j]
            x[i] /= matrix[i][i]

    return x

# Question 4: Implement LU Factorization for the following
# matrix and do the following:
# a.) Print out the matrix determinant
# b.) Print out the L matrix
# c.) Print out the U matrix
# [  1  1  0  3]
# |  2  1 -1  1|
# |  3 -1 -1  2|
# [ -1  2  3 -1]
def lu(matrix):
    n = len(matrix)
    l = np.zeros((n, n))
    u = np.zeros((n, n))

    for i in range(n):
        l[i][i] = 1

    u[0] = matrix[0]

    if l[0][0] * u[0][0] == 0:
        return -1

    for j in range(1, n):
        u[0][j] = matrix[0][j] / l[0][0]
        l[j][0] = matrix[j][0] / u[0][0]

    for i in range(1, n):
        u[i][i] = matrix[i][i]
        for k in range(i):
            u[i][i] -= l[i][k] * u[k][i]
        for j in range(i + 1, n):
            u[i][j] = matrix[i][j]
            l[j][i] = matrix[j][i]
            for k in range(i):
                u[i][j] -= l[i][k] * u[k][j]
                l[j][i] -= l[j][k] * u[k][i]
            if l[i][i] * u[i][i] == 0:
                return -1
            u[i][j] /= l[i][i]
            l[j][i] /= u[i][i]

    if l[n - 1][n - 1] * u[n - 1][n - 1] == 0:
        return -1
    det = 1
    for i in range(n):
        det *= u[i][i]
    return (det, l, u)

# Determine if the following matrix is diagonally dominant.
# [  9  0  5  2  1]
# |  3  9  1  2  1|
# |  0  1  7  2  3|
# |  4  2  3 12  2|
# [  3  2  4  0  8]
def diag(matrix):
    n = len(matrix)
    for i in range(n):
        tot = 0
        for j in range(n):
            if (j == i):
                continue
            tot += abs(matrix[i][j])
        if tot > matrix[i][i]:
            return False
    return True

# Determine if the matrix is a positive definite.
# [  2  2  1]
# |  2  3  0|
# [  1  0  2]
def posdef(matrix):
    if lu(matrix)[0] != 0:
        return True
    return False
    return 0

def main():
    ans1 = euler(dy, 0, 2, 10, 0, 1)
    print("%.5f" % ans1, end="\r\n\r\n")
    ans2 = runge_kutta(dy, 0, 2, 10, 0, 1)
    print("%.5f" % ans2, end="\r\n\r\n")
    matrix1 = np.zeros((3, 4))
    matrix1[0] = [2, -1, 1, 6]
    matrix1[1] = [1, 3, 1, 0]
    matrix1[2] = [-1, 5, 4, -3]
    ans3 = gaussian(matrix1)
    print(ans3, end="\r\n\r\n")
    matrix2 = np.zeros((4, 4))
    matrix2[0] = [1, 1, 0, 3]
    matrix2[1] = [2, 1, -1, 1]
    matrix2[2] = [3, -1, -1, 2]
    matrix2[3] = [-1, 2, 3, -1]
    ans4 = lu(matrix2)
    ans4a = ans4[0]
    ans4b = ans4[1]
    ans4c = ans4[2]
    print("%.5f" % ans4a, end="\r\n\r\n")
    print(ans4b, end="\r\n\r\n", sep="\r\n")
    print(ans4c, end="\r\n\r\n")
    matrix3 = np.zeros((5,5))
    matrix3[0] = [9, 0, 5, 2, 1]
    matrix3[1] = [3, 9, 1, 2, 1]
    matrix3[2] = [0, 1, 7, 2, 3]
    matrix3[3] = [4, 2, 3, 12, 2]
    matrix3[4] = [3, 2, 4, 0, 8]
    ans5 = diag(matrix3)
    print(ans5, end="\r\n\r\n")
    matrix4 = np.zeros((3, 3))
    matrix4[0] = [2, 2, 1]
    matrix4[1] = [2, 3, 0]
    matrix4[2] = [1, 0, 2]
    ans6 = posdef(matrix4)
    print(ans6, end="\r\n")
if __name__ == "__main__":
    main()

