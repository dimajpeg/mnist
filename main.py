# Імпорт необхідних бібліотек
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# 1. Завантаження набору даних MNIST
# load_data() повертає два кортежі: (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Аналіз структури завантажених даних
print("\nАналіз структури даних:")
print(f"Розмірність навчальних зображень (x_train): {x_train.shape}")
print(f"Розмірність міток навчальних зображень (y_train): {y_train.shape}")
print(f"Розмірність тестових зображень (x_test): {x_test.shape}")
print(f"Розмірність міток тестових зображень (y_test): {y_test.shape}")

print(f"\nТип даних зображень (x_train.dtype): {x_train.dtype}")
print(f"Тип даних міток (y_train.dtype): {y_train.dtype}")

print(f"\nПерші 5 міток з навчальної вибірки (y_train): {y_train[:5]}")
print(f"Перші 5 міток з тестової вибірки (y_test): {y_test[:5]}")

# Мінімальне та максимальне значення пікселів (для розуміння діапазону)
print(f"\nМінімальне значення пікселя в x_train: {np.min(x_train)}")
print(f"Максимальне значення пікселя в x_train: {np.max(x_train)}")

# 3. Візуалізація декількох прикладів зображень
print("\nВізуалізація прикладів зображень:")
num_images_to_show = 10 # Кількість зображень для візуалізації
plt.figure(figsize=(10, 2)) # Розмір фігури для візуалізації

for i in range(num_images_to_show):
    plt.subplot(1, num_images_to_show, i + 1) # Створюємо підграфік 1xN
    plt.imshow(x_train[i], cmap='gray') # Відображаємо i-те зображення в відтінках сірого
    plt.title(f"Мітка: {y_train[i]}") # Встановлюємо заголовок з міткою класу
    plt.axis('off') # Вимикаємо осі для кращого вигляду

plt.suptitle("Приклади зображень з набору даних MNIST (навчальна вибірка)") # Загальний заголовок
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Для кращого розташування підписів
plt.show() # Показати фігуру

# (Опціонально) Можна зберегти фігуру для вставки в курсову
# plt.savefig("mnist_examples.png")
# print("Приклади зображень збережено у файл mnist_examples.png")