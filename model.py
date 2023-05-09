import tensorflow as tf
from tensorflow.keras import layers

def build_model(input_shape, num_keys):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_keys, activation="softmax")
    ])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))
