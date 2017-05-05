import numpy as np


def load_vectors(word_index, num_words=None):
    if num_words is None:
        num_words = len(word_index) + 1
    # Prepare embedding
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open("/Users/lambrosini/PycharmProjects/IberEval2017/resources/word2vec/es.vec")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    print('Preparing embedding matrix.')

    # prepare embedding matrix
    embedding_matrix = np.zeros((num_words, 300))
    found, oob = 0, 0
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            found += 1
        else:
            oob += 1
    print("Vectors embedding summary: {found} token found, {oob} Out Of Vocabulary".format(found=found, oob=oob))
    return embedding_matrix
