import torch
import numpy as np
import ROOT
ROOT.EnableThreadSafety()

tree_name = "tree_name"
file_name = f"path_to_file"

chunk_size = 100_000
batch_size = 1024

target = "target_name"

num_columns = 0  # set number of columns

gen_train, gen_validation = ROOT.TMVA.Experimental.CreatePyTorchDataLoaders(
    tree_name, file_name, batch_size, chunk_size, target=target, validation_split=0.3)


def calc_accuracy(targets, pred):
    return torch.sum(targets == pred.round()) / pred.size(0)


# Initialize PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(num_columns - 1, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 1),
    torch.nn.Sigmoid()
)
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# Loop through the training set and train model
for i, (x_train, y_train) in enumerate(gen_train):
    # Make prediction and calculate loss
    pred = model(x_train).view(-1)
    loss = loss_fn(pred, y_train)

    # improve model
    model.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    accuracy = calc_accuracy(y_train, pred)

    print(f"Training => {accuracy = }")

print(f"Start Validation")

# Evaluate the model on the validation set
for i, (x_train, y_train) in enumerate(gen_validation):
    # Make prediction and calculate loss
    pred = model(x_train).view(-1)
    loss = loss_fn(pred, y_train)

    # Calculate accuracy
    accuracy = calc_accuracy(y_train, pred)

    print(f"Validation => {accuracy = }")
