import torch
import torch.nn.functional as F
import networkx as nx
import dgl
from dgl.nn.pytorch.conv import SAGEConv
import numpy as np

# Verify NumPy version
print(f"NumPy version: {np.__version__}")

# Load the GraphML file
try:
    nx_graph = nx.read_graphml('C:\\Users\\xxbla\\OneDrive\\Documents\\VSCode\\CETA\\CETANN\\Usman\\withoutEdge.graphml')
except Exception as e:
    print(f"Error loading GraphML file: {e}")
    raise

# Inspect node attributes to understand available features
print("Node attributes:")
for node, data in nx_graph.nodes(data=True):
    print(data)
    break  # print only the first node's attributes for inspection

# Preprocess node attributes to handle non-numeric values
def preprocess_attributes(graph, attrs):
    for attr in attrs:
        values = nx.get_node_attributes(graph, attr)
        if not values:
            continue
        sample_value = next(iter(values.values()))
        if isinstance(sample_value, str):
            unique_values = list(set(values.values()))
            value_mapping = {val: idx for idx, val in enumerate(unique_values)}
            nx.set_node_attributes(graph, {node: value_mapping.get(val, -1) for node, val in values.items()}, attr)
        elif sample_value is None or sample_value == '':
            nx.set_node_attributes(graph, {node: -1 for node in values.keys()}, attr)

node_attrs = ['ID_RSSD', 'labels', 'BHC_IND', 'BROAD_REG_CD', 'CHTR_AUTH_CD', 'CHTR_TYPE_CD']  # Add more relevant attributes
optional_attrs = ['train_mask', 'val_mask', 'test_mask']

# Preprocess all node attributes to ensure they are numeric
preprocess_attributes(nx_graph, node_attrs + optional_attrs)

# Check if optional attributes are present in the graph
for attr in optional_attrs:
    if any(attr in data for node, data in nx_graph.nodes(data=True)):
        node_attrs.append(attr)

# Convert NetworkX graph to DGL graph
try:
    g = dgl.from_networkx(nx_graph, node_attrs=node_attrs)
except Exception as e:
    print(f"Error converting to DGL graph: {e}")
    raise

# Use multiple features
feature_keys = ['ID_RSSD', 'BHC_IND', 'BROAD_REG_CD', 'CHTR_AUTH_CD', 'CHTR_TYPE_CD']  # List of features

# Ensure node features are available
try:
    features = torch.tensor([[float(data[key]) for key in feature_keys] for node, data in nx_graph.nodes(data=True)], dtype=torch.float32)
    g.ndata['feat'] = features
except KeyError as e:
    print(f"Feature key error: {e}")
    raise

# Map string labels to integers for classification
label_mapping = {}
current_label = 0

# Assuming 'labels' attribute is used for node classification
try:
    labels_list = []
    for node, data in nx_graph.nodes(data=True):
        label = data['labels']
        if label not in label_mapping:
            label_mapping[label] = current_label
            current_label += 1
        labels_list.append(label_mapping[label])
    labels = torch.LongTensor(labels_list)
except KeyError as e:
    print(f"Label key error: {e}")
    raise

# Define the GraphSAGE model
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, 'mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, 'mean')

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = self.conv2(g, x)
        return F.log_softmax(x, dim=1)

# Instantiate the model, define the optimizer and loss function
model = GraphSAGEModel(g.ndata['feat'].shape[1], 16, len(set(labels.numpy())))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Split data into train, validation, and test sets (handling missing attributes)
def create_mask(attr_name):
    if attr_name in g.ndata:
        return g.ndata[attr_name].bool()
    else:
        return torch.zeros(g.number_of_nodes(), dtype=torch.bool)

train_mask = create_mask('train_mask')
val_mask = create_mask('val_mask')
test_mask = create_mask('test_mask')

# Ensure masks are not empty
if train_mask.sum().item() == 0:
    train_mask[0] = True
if val_mask.sum().item() == 0:
    val_mask[0] = True
if test_mask.sum().item() == 0:
    test_mask[0] = True

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(g, g.ndata['feat'])
    loss = criterion(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
def test(mask):
    model.eval()
    out = model(g, g.ndata['feat'])
    pred = out.argmax(dim=1)
    correct = pred[mask] == labels[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc, pred

# Load data and train the model
try:
    for epoch in range(200):
        loss = train()
        if epoch % 10 == 0:
            acc, _ = test(test_mask)
            print(f'Epoch {epoch}, Loss: {loss}, Test Accuracy: {acc}')
except RuntimeError as e:
    print(f"Error during training/testing: {e}")
    raise

# Get predictions for the entire dataset
try:
    model.eval()
    out = model(g, g.ndata['feat'])
    pred = out.argmax(dim=1)

    # Print detailed predictions
    for node_id, prediction in enumerate(pred):
        print(f"Node {node_id} (Paper ID): Predicted Class: {prediction.item()}")

    # Optionally, return predictions
    def get_predictions():
        model.eval()
        out = model(g, g.ndata['feat'])
        return out.argmax(dim=1)

    # Get and print predictions
    predictions = get_predictions()
    for node_id, prediction in enumerate(predictions):
        print(f"Node {node_id} (Paper ID): Predicted Class: {prediction.item()}")
except RuntimeError as e:
    print(f"Error getting predictions: {e}")
    raise
