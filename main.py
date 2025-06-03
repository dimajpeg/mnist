# Імпорт необхідних бібліотек
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# 1. Завантаження набору даних MNIST
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = mnist.load_data()

# Зробимо копії, щоб оригінальні дані для візуалізації не змінювалися
x_train = np.copy(x_train_orig)
y_train = np.copy(y_train_orig)
x_test = np.copy(x_test_orig)
y_test = np.copy(y_test_orig)


# 2. Аналіз структури завантажених даних (на оригінальних даних)
print("\nАналіз структури завантажених даних (до обробки):")
print(f"Розмірність навчальних зображень (x_train_orig): {x_train_orig.shape}")
print(f"Розмірність міток навчальних зображень (y_train_orig): {y_train_orig.shape}")
print(f"Розмірність тестових зображень (x_test_orig): {x_test_orig.shape}")
print(f"Розмірність міток тестових зображень (y_test_orig): {y_test_orig.shape}")

print(f"\nТип даних зображень (x_train_orig.dtype): {x_train_orig.dtype}")
print(f"Тип даних міток (y_train_orig.dtype): {y_train_orig.dtype}")

print(f"\nПерші 5 міток з навчальної вибірки (y_train_orig): {y_train_orig[:5]}")
print(f"Перші 5 міток з тестової вибірки (y_test_orig): {y_test_orig[:5]}")

print(f"\nМінімальне значення пікселя в x_train_orig: {np.min(x_train_orig)}")
print(f"Максимальне значення пікселя в x_train_orig: {np.max(x_train_orig)}")

# 3. Візуалізація декількох прикладів зображень (з ОРИГІНАЛЬНИХ даних)
# print("\nВізуалізація прикладів зображень (до обробки):")
# num_images_to_show = 10
# plt.figure(figsize=(10, 2))

# for i in range(num_images_to_show):
#     plt.subplot(1, num_images_to_show, i + 1)
#     plt.imshow(x_train_orig[i], cmap='gray')
#     plt.title(f"Мітка: {y_train_orig[i]}")
#     plt.axis('off')

# plt.suptitle("Приклади зображень з набору даних MNIST (навчальна вибірка, до обробки)")
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# # plt.savefig("mnist_examples_before_processing.png")
# # plt.show() # Закоментовано, щоб не блокувати виконання при кожному запуску


# 4. ПОПЕРЕДНЯ ОБРОБКА ДАНИХ
print("\nПопередня обробка даних...")

# 4.1. Зміна форми вхідних зображень
img_rows, img_cols = x_train.shape[1], x_train.shape[2] # 28, 28
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channels)
input_shape = (img_rows, img_cols, num_channels)

print(f"Нова розмірність навчальних зображень (x_train після reshape): {x_train.shape}")
print(f"Нова розмірність тестових зображень (x_test після reshape): {x_test.shape}")
print(f"Форма вхідних даних для моделі (input_shape): {input_shape}")

# 4.2. Нормалізація значень пікселів
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

print(f"\nТип даних зображень після нормалізації (x_train.dtype): {x_train.dtype}")
print(f"Мінімальне значення пікселя в x_train після нормалізації: {np.min(x_train):.4f}")
print(f"Максимальне значення пікселя в x_train після нормалізації: {np.max(x_train):.4f}")

# 4.3. Перетворення міток класів у категоріальний формат (One-Hot Encoding)
num_classes = 10

print(f"\nПерші 5 міток y_train до One-Hot Encoding: {y_train_orig[:5]}") # Показуємо оригінальні для порівняння
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"Розмірність міток y_train після One-Hot Encoding: {y_train.shape}")
print(f"Перша мітка y_train[0] після One-Hot Encoding (цифра {y_train_orig[0]}): {y_train[0]}")

print("\nПопередня обробка даних завершена.")


# 5. ФОРМУВАННЯ МОДЕЛІ НЕЙРОННОЇ МЕРЕЖІ
print("\nФормування моделі нейронної мережі...")

model = Sequential()

# Перший згортковий блок
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Другий згортковий блок
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Шар вирівнювання
model.add(Flatten())

# Повнозв'язний шар
model.add(Dense(128, activation='relu'))

# Шар Dropout для регуляризації
model.add(Dropout(0.5))

# Вихідний повнозв'язний шар
model.add(Dense(num_classes, activation='softmax'))

# Виведемо структуру створеної моделі
print("\nСтруктура моделі:")
model.summary()

print("\nФормування моделі завершено.")


# 6. КОМПІЛЯЦІЯ МОДЕЛІ НЕЙРОННОЇ МЕРЕЖІ
print("\nКомпіляція моделі...")

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nКомпіляція моделі завершена.")


# 7. ТРЕНУВАННЯ МОДЕЛІ НЕЙРОННОЇ МЕРЕЖІ
# (Включає визначення параметрів навчання)
print("\nВизначення параметрів та тренування моделі...")

# Визначення параметрів партії навчальних даних та кількості епох
batch_size = 128
epochs = 15 # Можеш змінити на 2, якщо викладач наполягає на "початковому значенні" з таблиці,
            # але для демонстрації навчання 15 краще. Або почни з 2, а потім збільш.

print(f"Параметри навчання: batch_size = {batch_size}, epochs = {epochs}")

# Тренування моделі нейронної мережі
# Метод fit повертає об'єкт History, який містить історію навчання
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, # verbose=1 показуватиме прогрес-бар навчання
                    validation_data=(x_test, y_test)) # Використовуємо тестові дані для валідації

print("\nТренування моделі завершено.")
# Об'єкт 'history' тепер містить дані про втрати та точність на кожній епосі
# для навчальної та валідаційної вибірок.
# Ми використаємо 'history' на наступних етапах для візуалізації результатів навчання.

# Наступний етап - оцінка моделі та візуалізація
# score = model.evaluate(x_test, y_test, verbose=0)
# print(f"\nTest loss: {score[0]}")
# print(f"Test accuracy: {score[1]}")