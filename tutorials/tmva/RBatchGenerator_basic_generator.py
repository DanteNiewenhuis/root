import time
import numpy as np
import ROOT

ROOT.EnableThreadSafety()

tree_name = "tree_name"
file_name = "path_to_file"

chunk_size = 1_000_000
batch_size = 100_000

ds_train, ds_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
    tree_name, file_name, batch_size, chunk_size, validation_split=0.3
)

# Loop through training set
for i, b in enumerate(ds_train):
    print(f"Training batch {i} => {len(b) = }")

# Loop through Validation set
print("Starting Validation")
for i, b in enumerate(ds_validation):
    print(f"Validation batch {i} => {len(b) = }")
