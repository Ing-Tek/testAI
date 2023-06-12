import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv('gpascore.csv')
data = data.dropna()

y = data['admit'].values
x = data.drop(['admit'], axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_normalized = scaler.fit_transform(x_train)
x_test_normalized = scaler.transform(x_test)

max_reruns = 3
rerun_count = 0
desired_accuracy = 0.8
current_accuracy = 0.0
max_accuracy = 0.0

while current_accuracy <= desired_accuracy and rerun_count < max_reruns:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dense(256, activation='tanh'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='accuracy', patience=800, restore_best_weights=True)
    model.fit(x_train_normalized, y_train, epochs=1000, batch_size=64, validation_data=(x_test_normalized, y_test), callbacks=[early_stopping])

    for i in model.history.history['accuracy']:
        if i > max_accuracy:
            max_accuracy = i

    current_accuracy = model.history.history['accuracy'][-1]
    print(current_accuracy)
    rerun_count += 1

if current_accuracy <= desired_accuracy:
    print("Failed to reach the desired accuracy. Try again.")
    exit(1)

print("The maximum accuracy reached was " + str(max_accuracy) + ".")

print("Reached the desired accuracy after " + str(rerun_count - 1) + " reruns.")
p = model.predict([[750, 3.50, 1], [100, 2.2, 1]])
print(p)