import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GraphSAGE
import numpy as np

# Verify NumPy version
print(f"NumPy version: {np.__version__}")

# Load the Cora dataset
try:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
except RuntimeError as e:
    print(f"Error loading dataset: {e}")
    raise

# Print dataset statistics
data = dataset[0]
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of node features: {data.num_node_features}")
print(f"Number of edge features: {data.num_edge_features}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Train mask sum: {data.train_mask.sum()}")
print(f"Validation mask sum: {data.val_mask.sum()}")
print(f"Test mask sum: {data.test_mask.sum()}")

# Define the GraphSAGE model
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = GraphSAGE(in_channels, hidden_channels, num_layers=2)
        self.conv2 = GraphSAGE(hidden_channels, out_channels, num_layers=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Instantiate the model, define the optimizer and loss function
model = GraphSAGEModel(dataset.num_node_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
    return acc, pred

# Load data and train the model
try:
    data = dataset[0]
    for epoch in range(200):
        loss = train()
        if epoch % 10 == 0:
            acc, _ = test()
            print(f'Epoch {epoch}, Loss: {loss}, Test Accuracy: {acc}')
except RuntimeError as e:
    print(f"Error during training/testing: {e}")
    raise

# Get predictions for the entire dataset
try:
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    print("Predictions:", pred)

    # Optionally, return predictions
    def get_predictions():
        model.eval()
        out = model(data)
        return out.argmax(dim=1)

    # Get and print predictions
    predictions = get_predictions()
    print("Predictions:", predictions)
except RuntimeError as e:
    print(f"Error getting predictions: {e}")
    raise


#see if graph ml is supported by graph sage
#also want to be able to have entity predictions