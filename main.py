import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Загрузка данных
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Нормализация данных и преобразование в формат, подходящий для обработки в нейронной сети
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Определение модели
model = keras.Sequential(
    [
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

# Компиляция модели
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Обучение модели
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)

# Оценка точности модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
