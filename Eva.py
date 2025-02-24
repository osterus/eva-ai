import tensorflow as tf
import numpy as np
import random

# Генерация данных для тренировки
def generate_data(num_samples):
    x_train = []
    y_train = []
    for _ in range(num_samples):
        a = random.randint(1, 10000000)
        b = random.randint(1, 10000000)
        x_train.append([a, b])  # Вход: два числа
        y_train.append(a + b)  # нкобзодимый выход: их сумма
    return np.array(x_train), np.array(y_train)

# Генерация тренировочного масива (dataset)
x_train, y_train = generate_data(10000000)  # 100,000 примеров

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),  # 2 входа a и b
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')  # Выход: одно число (сумма)
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='mean_squared_error',  # Функция потерь для регрессии
              metrics=['mean_absolute_error'])  # Метрика

# Обучение модели
model.fit(x_train, y_train, epochs=5, batch_size=25600)

#  сохранение модели
def save_model(model, filename='my_model'):
    # Сохранение модели в формате .h5
    model.save(filename + '.h5')
    print(f"Модель сохранена в {filename}.h5")

# Проверка модели
test_input = np.array([[5, 7]])  # Пример: 5 + 7
predicted_sum = model.predict(test_input)

print(f"Модель предсказала: {predicted_sum[0][0]:.2f}")
print(f"Правильный ответ: {5 + 7}")
x=' ' # иначе не хочут работать линия 46
#использование
while x != 'y':
    c=int(input())
    v=int(input())
    test_input = np.array([[c, v]])  # Пример: 5 + 7
    predicted_sum = model.predict(test_input)
    print(f"Модель предсказала: {predicted_sum[0][0]:.2f}")
    print(f"Правильный ответ: {c + v}")
    print(f"ошибка: {(c + v)-(predicted_sum)}")
    x=input('stop?')
    if x == 'save':
        # Сохранить модель после тренировки
        save_model(model, 'sum_model')