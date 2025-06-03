# Імпорт необхідних бібліотек
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix # Для звіту та матриці помилок
import seaborn as sns # Для візуалізації матриці помилок

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# 1. Завантаження набору даних MNIST
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = mnist.load_data()

# Зберігаємо копії оригінальних міток (0-9) для фінального звіту
y_train_labels = np.copy(y_train_orig)
y_test_labels = np.copy(y_test_orig)

# Робочі копії даних, які будуть змінюватися
x_train = np.copy(x_train_orig)
x_test = np.copy(x_test_orig)


# 2. Аналіз структури завантажених даних (на оригінальних даних)
print("\nАналіз структури завантажених даних (до обробки):")
print(f"Розмірність навчальних зображень (x_train_orig): {x_train_orig.shape}")
print(f"Розмірність міток навчальних зображень (y_train_orig): {y_train_orig.shape}")
print(f"Розмірність тестових зображень (x_test_orig): {x_test_orig.shape}")
print(f"Розмірність міток тестових зображень (y_test_orig): {y_test_orig.shape}")
# ... (інші print для аналізу)


# 3. Візуалізація декількох прикладів зображень (з ОРИГІНАЛЬНИХ даних)
print("\nВізуалізація прикладів зображень (до обробки):")
num_images_to_show = 10
plt.figure(figsize=(10, 2))
for i in range(num_images_to_show):
    plt.subplot(1, num_images_to_show, i + 1)
    plt.imshow(x_train_orig[i], cmap='gray')
    plt.title(f"Мітка: {y_train_orig[i]}")
    plt.axis('off')
plt.suptitle("Приклади зображень з набору даних MNIST (навчальна вибірка, до обробки)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("mnist_examples_before_processing.png")
plt.close() # Закриваємо, щоб не блокувати


# 4. ПОПЕРЕДНЯ ОБРОБКА ДАНИХ
print("\nПопередня обробка даних...")
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
num_channels = 1
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channels)
input_shape = (img_rows, img_cols, num_channels)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
num_classes = 10
# Для навчання та оцінки моделі використовуємо мітки у форматі one-hot
y_train_categorical = tf.keras.utils.to_categorical(y_train_labels, num_classes)
y_test_categorical = tf.keras.utils.to_categorical(y_test_labels, num_classes)
print("Попередня обробка даних завершена.")


# 5. ФОРМУВАННЯ МОДЕЛІ НЕЙРОННОЇ МЕРЕЖІ
print("\nФормування моделі нейронної мережі...")
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
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
print("\nВизначення параметрів та тренування моделі...")
batch_size = 128
epochs = 15
print(f"Параметри навчання: batch_size = {batch_size}, epochs = {epochs}")
history = model.fit(x_train, y_train_categorical,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test_categorical))
print("\nТренування моделі завершено.")


# 8. ОЦІНКА МОДЕЛІ НА ТЕСТОВОМУ НАБОРІ ТА ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ НАВЧАННЯ
print("\nОцінка моделі на тестовому наборі...")
score = model.evaluate(x_test, y_test_categorical, verbose=0)
print(f"\nПідсумкові втрати на тестовій вибірці (Test loss): {score[0]:.4f}")
print(f"Підсумкова точність на тестовій вибірці (Test accuracy): {score[1]:.4f}")

print("\nВізуалізація результатів навчання...")
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Втрати на навчанні (Training Loss)')
plt.plot(history.history['val_loss'], label='Втрати на валідації (Validation Loss)')
plt.title('Динаміка функції втрат під час навчання')
plt.xlabel('Епохи')
plt.ylabel('Втрати (Loss)')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Точність на навчанні (Training Accuracy)')
plt.plot(history.history['val_accuracy'], label='Точність на валідації (Validation Accuracy)')
plt.title('Динаміка точності під час навчання')
plt.xlabel('Епохи')
plt.ylabel('Точність (Accuracy)')
plt.legend()
plt.grid(True)
plt.suptitle('Результати навчання моделі CNN на наборі даних MNIST', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("training_history_loss_accuracy.png")
plt.close() # Закриваємо, щоб не блокувати


# 9. ПРОГНОЗУВАННЯ НОМЕРІВ КЛАСІВ
print("\nПрогнозування номерів класів...")
y_pred_probabilities = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probabilities, axis=1)

# Виведемо перші декілька істинних та прогнозованих класів для перевірки
# y_test_labels - це наші оригінальні мітки (0-9), які ми зберегли на початку
print(f"\nПерші 10 істинних класів (з тестової вибірки): {y_test_labels[:10]}")
print(f"Перші 10 прогнозованих класів:               {y_pred_classes[:10]}")


# 10. ЗВІТ ПРО ЯКІСТЬ КЛАСИФІКАЦІЇ ДЛЯ КЛАСІВ
print("\nЗвіт про якість класифікації для кожного класу:")
target_names = [str(i) for i in range(num_classes)]
report = classification_report(y_test_labels, y_pred_classes, target_names=target_names)
print(report)

print("\nМатриця помилок (Confusion Matrix) - текстовий вивід:")
conf_matrix = confusion_matrix(y_test_labels, y_pred_classes)
print(conf_matrix)

print("\nВізуалізація матриці помилок...")
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Прогнозовані класи')
plt.ylabel('Істинні класи')
plt.title('Матриця помилок')
plt.savefig("confusion_matrix.png")
plt.close() # Закриваємо, щоб не блокувати

print("\nПрогнозування та звіт про якість завершені. Роботу скрипта завершено.")