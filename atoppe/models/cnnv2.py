import keras
from keras.engine import Input, Model
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import Dense
from keras.layers import Embedding
from keras.preprocessing import sequence

from models.mymodel import MyModel


class CNNModelv2(MyModel):
    def build(self, params):
        self.x_train = sequence.pad_sequences(self.x_train, maxlen=params['maxlen'])
        self.x_test = sequence.pad_sequences(self.x_test, maxlen=params['maxlen'])

        x = Input(shape=(params['maxlen'],))
        emb = Embedding(params['max_features'],
                        params['embedding_dims'],
                        input_length=params['maxlen'])(x)
        merge_input = []
        for kernel_size in [2, 3, 4]:
            conv = Conv1D(filters=params['filters'],
                          kernel_size=kernel_size,
                          padding=params['padding'],
                          dilation_rate=params['dilation_rate'],
                          activation='relu', input_shape=(params['maxlen'], params['embedding_dims']))(emb)
            max_pooling = MaxPooling1D(pool_size=params['pool_size'])(conv)
            flatten = Flatten()(max_pooling)
            merge_input.append(flatten)

        merged = keras.layers.concatenate(merge_input)
        y = Dense(self.output_size, activation='sigmoid')(merged)
        self.keras_model = Model(inputs=x, outputs=y)

        self.keras_model.compile(loss='categorical_crossentropy',
                                 optimizer='adam', metrics=params['metrics'])
