from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

# Dummy Data
data = np.random.random((1000, 784))
labels = np.random.randint(10, size=(1000, 1))
labels = to_categorical(labels, 10)

inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optiomizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)
