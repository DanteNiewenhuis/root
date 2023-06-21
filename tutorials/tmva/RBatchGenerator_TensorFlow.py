import tensorflow as tf
import ROOT
ROOT.EnableThreadSafety()

tree_name = "tree_name"
file_name = "path_to_file"

batch_size = 1024
chunk_size = 1_000_000

target = "target"

num_columns = 0  # set number of columns

ds_train, ds_valid = ROOT.TMVA.Experimental.CreateTFDatasets(tree_name, file_name, batch_size, chunk_size,
                                                             validation_split=0.3, target="Type")


###################################################################################################
# AI example
###################################################################################################

# Define TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(300, activation=tf.nn.tanh,
                          input_shape=(num_columns - 1,)),  # input shape required
    tf.keras.layers.Dense(300, activation=tf.nn.tanh),
    tf.keras.layers.Dense(300, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

loss_fn = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train the model
model.fit(ds_train, validation_data=ds_valid, epochs=2)
