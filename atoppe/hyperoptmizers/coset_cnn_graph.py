import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

from data_loaders import coset

# Fixed params

(x_train, y_train), (x_test, y_test) = coset.load_data()
x_train = sequence.pad_sequences(x_train, maxlen=50)
x_test = sequence.pad_sequences(x_test, maxlen=50)

max_len = 50
# Parameter that need to be optimized

max_features = 15000
batch_size = 40
embedding_dims = 50
filters = 100
kernel_size = 3
hidden_dims = 150
epochs = 10
drop_out_chance_embedding = 0.3
drop_out_chance_dense = 0.3

model = Sequential()

model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=max_len))
model.add(Dropout(drop_out_chance_embedding))

model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_dims))
model.add(Dropout(drop_out_chance_dense))
model.add(Activation('relu'))

model.add(Dense(5))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])
print(model.metrics_names)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test))

model.evaluate(x_test, y_test,
               batch_size=batch_size)

f, axarr = plt.subplots(2, sharex=True)
# Categorical accuracy
axarr[0].plot(history.history['categorical_accuracy'])
axarr[0].plot(history.history['val_categorical_accuracy'])
axarr[0].set_title('Categorical accuracy')
axarr[0].legend(['train', 'test'], loc='upper left')
# Loss function value
axarr[1].plot(history.history['loss'])
axarr[1].plot(history.history['val_loss'])
axarr[1].set_title('Categorical cross-entropy')
axarr[1].legend(['train', 'test'], loc='upper left')

plt.show()
