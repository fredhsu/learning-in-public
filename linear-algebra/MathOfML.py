from csv import QUOTE_ALL
import numpy as np
from timeit import timeit


def mse(x: np.ndarray, y: np.ndarray) -> float:
    n = float(len(x))
    return sum([(x - y) ** 2 for (x, y) in zip(x, y)]) / n


def np_mse(x: np.ndarray, y: np.ndarray):
    return np.mean(np.square(x - y))


def weighted_pnorm(w, x, p) -> float:
    return np.sum(w * np.abs(x) ** p) ** (1 / p)


def cos_similarity(x, y):
    return x / np.linalg.norm(x) @ y / np.linalg.norm(y)


def fib_recur(n):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        return fib_recur(n - 1) + fib_recur(n - 2)


def fib_mat(n):
    x = np.array([[1, 1], [1, 0]])
    result = x
    if n <= 1:
        return x
    else:
        for _ in range(0, n):
            result = result @ x
    return result[1][0]


def hadamard_loop(a, b):
    n = a.shape[0]
    m = a.shape[1]
    result = np.zeros_like(a)
    for i in range(n):
        for j in range(m):
            result[i, j] = a[i][j] * b[i][j]
    return result


def bilinear(a, x, y):
    return x.T @ a @ y


def main():
    # Problem 1
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    print(mse(x, y))
    print(np_mse(x, y))

    # Problem 2
    n_runs = 100000
    size = 38
    t_naive = timeit(
        "max(x)",
        setup=f"import numpy as np; rng = np.random.default_rng(); x = rng.random({size})",
        number=n_runs,
    )
    t_numpy = timeit(
        "np.max(x)",
        setup=f"import numpy as np; rng = np.random.default_rng(); x = rng.random({size})",
        number=n_runs,
    )
    print(t_naive)
    print(t_numpy)
    # breakeven around 38/39 items

    # Problem 4
    print(weighted_pnorm(x, y, 2))

    # Problem 5
    print(f"cos_similarity: {cos_similarity(x, x)} -- {cos_similarity(x, y)}")

    # Problem 6
    a = np.array([[-1, 2], [1, 5]])
    b = np.array([[6, -2], [2, -6], [-3, 2]])
    print(f"a. {b @ a}")
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8], [9, 10]])
    print(f"b. {b @ a}")

    # problem 7
    #

    t_recur = timeit(
        "fib_recur(10)",
        setup="from __main__ import fib_recur",
        number=n_runs,
    )
    t_mat = timeit(
        "fib_mat(10)",
        setup="from __main__ import fib_mat",
        number=n_runs,
    )
    print(f"recur:{t_recur}")
    print(f"mat:{t_mat}")

    # problem 8
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[4, 5, 6], [7, 8, 9]])
    print(f"hadamard_loop: {hadamard_loop(a, b)}")
    print(f"hadamard_numpy: {a * b}")
    # problem 9

    a = np.array([[1, 2], [3, 4]])
    x = np.array([5, 6])
    y = np.array([7, 8])
    print(f"bilinear: {bilinear(a, x, y)}")


if __name__ == "__main__":
    main()
