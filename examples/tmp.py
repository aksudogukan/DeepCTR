import tensorflow as tf

# Define input sparse matrix
input_sparse_matrix = tf.SparseTensor(
    indices=[[0, 1], [1, 2], [2, 0], [3, 3]],
    values=[1, 1, 1, 1],
    dense_shape=[4, 4])

# Define embedding size and dimension
embedding_size = 4
embedding_dim = 10

# Define embedding layer
embedding_layer = tf.keras.layers.Embedding(
    input_dim=input_sparse_matrix.dense_shape[1],
    output_dim=embedding_dim,
    input_length=1,
    name="embedding_layer"
)

# Flatten the sparse matrix and apply the embedding layer
flattened_matrix = tf.keras.layers.Flatten()(input_sparse_matrix)
embedded_matrix = embedding_layer(flattened_matrix)

# Print the resulting embedding vector
print(embedded_matrix)
