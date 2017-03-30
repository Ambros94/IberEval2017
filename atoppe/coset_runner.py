from data_loaders import coset
from models.cnn import CNNModel

cnn = CNNModel(data_function=coset.load_data)
score, acc = cnn.run(max_features=15000, maxlen=50, batch_size=32, embedding_dims=50, filters=250, kernel_size=3,
                     hidden_dims=250, epochs=5)
print('Test score:', score)
print('Test accuracy:', acc)
