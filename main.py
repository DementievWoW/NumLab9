import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return np.exp(-x) * y

# Точное решение
def exact_solution(x):
    return np.exp(-np.exp(-x) + 1)

# Метод Адамса 2-го порядка
def adams_2nd_order(x0, y0, h, n):
    x = np.zeros(n)
    y = np.zeros(n)
    x[0], y[0] = x0, y0

    # Первый шаг методом Эйлера
    x[1] = x[0] + h
    y[1] = y[0] + h * f(x[0], y[0])

    # Метод Адамса
    for i in range(1, n - 1):
        x[i + 1] = x[i] + h
        y[i + 1] = y[i] + (h / 2) * (3 * f(x[i], y[i]) - f(x[i - 1], y[i - 1]))

    return x, y

x0, y0 = 0, 1
l = 5  # Длина отрезка
n_values = [100, 1000, 10000]  # Количество шагов

# Построение графиков
plt.figure(figsize=(12, 8))

for n in n_values:
    h = l / n
    x, y = adams_2nd_order(x0, y0, h, n)
    plt.plot(x, y, label=f"Численное решение (n={n})")

# Точное решение
x_exact = np.linspace(x0, x0 + l, 1000)
y_exact = exact_solution(x_exact)
plt.plot(x_exact, y_exact, label="Точное решение", linestyle="--", linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Сравнение точного и численного решений")
plt.legend()
plt.grid()
plt.show()