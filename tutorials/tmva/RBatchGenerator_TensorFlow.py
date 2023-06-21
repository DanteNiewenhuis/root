import tensorflow as tf
import ROOT

ROOT.EnableThreadSafety()

tree_name = "tree_name"
file_name = "path_to_file"

batch_size = 1024
chunk_size = 1_000_000

target = "target"

ds_train, ds_valid = ROOT.TMVA.Experimental.CreateTFDatasets(
    tree_name,
    file_name,
    batch_size,
    chunk_size,
    validation_split=0.3,
    target=target,
)

input_columns = ds_train.train_columns
num_features = len(input_columns)

##############################################################################
# AI example
##############################################################################

# Define TensorFlow model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            300, activation=tf.nn.tanh, input_shape=(num_features,)
        ),  # input shape required
        tf.keras.layers.Dense(300, activation=tf.nn.tanh),
        tf.keras.layers.Dense(300, activation=tf.nn.tanh),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ]
)

loss_fn = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# Train the model
model.fit(ds_train, validation_data=ds_valid, epochs=2)
