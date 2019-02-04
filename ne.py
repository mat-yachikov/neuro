# библиотека мат. функций для работы с массивами
import numpy as np

# матрица истинности XOR
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1, 1, 0]]).T  # T - транспонирование

# инициализируем генератор случайных чисел
np.random.seed(1)

# матрица весов нейросети заполняется случайными величинами:
weights1 = np.random.random((2, 2))
weights2 = np.random.random((2, 1))


# функция активации
def nonlin(x, deriv=False):
    # линейная
    if(deriv is True):
        return x*(1-x)
    # нелинейная
    return 1/(1+np.exp(-x))


for iter in range(1000000):
    z1 = np.dot(x, weights1)
    # на выходе первого слоя
    a1 = nonlin(z1)

    z2 = np.dot(a1, weights2)
    # на выходе второго слоя
    a2 = nonlin(z2)

    # отклонение
    error = y - a2
    # при достижении определенной точности - выход
    if (max(abs(error)) < 0.05):
        break
    
    # вычисляем новые состояния нейронов
    delta2 = error * nonlin(a2, deriv=True)
    l1error = delta2.dot(weights2.T)
    delta1 = l1error * nonlin(a1, deriv=True)

    weights2 += np.dot(a1.T, delta2)
    weights1 += np.dot(x.T, delta1)

print("result:")
print(a2)
print("iterations:")
print(iter)
