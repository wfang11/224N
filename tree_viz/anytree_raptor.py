from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter
import json

def wrap_text(text, max_words=5):
    words = text.split()
    wrapped_text = ""
    for i in range(0, len(words), max_words):
        wrapped_text += " ".join(words[i:i+max_words]) + "\n"
    return wrapped_text.strip()

# Load the tree data from a JSON file
with open("trees/GPT_1000_node_data.json", "r") as f:
    data = json.loads(f.read())

# Filter out layer 0 nodes and adjust children lists
filtered_data = {}
for key, value in data.items():
    if int(value['Layer']) > 0:  # Include only nodes above layer 0
        filtered_data[key] = value

# Clear children for nodes that were originally at layer 1
for key, value in filtered_data.items():
    if int(value['Layer']) == 1:
        value['Children'] = []  # Make layer 1 nodes leaf nodes

# Determine the maximum layer among all remaining nodes
MAX_LAYER = max(int(node['Layer']) for node in filtered_data.values())

# Identify nodes that are located at the maximum layer to set them as children of the new root node
max_layer_nodes = [int(key) for key, value in filtered_data.items() if value['Layer'] == MAX_LAYER]

# Add a new root node with children that are the nodes from the maximum layer
filtered_data['-1'] = {
    'Node': -1,
    'Layer': MAX_LAYER + 1,
    'Text': 'ROOT',
    'Children': max_layer_nodes
}

# Initialize nodes with wrapped text and no parents from the filtered data
nodes = {int(key): Node(name=wrap_text(filtered_data[key]['Text'], max_words=6)) for key in filtered_data}

# Establish parent-child relationships based on the 'Children' list in the filtered data
for key, value in filtered_data.items():
    for child in value['Children']:
        nodes[child].parent = nodes[int(key)]

# Configure the tree visualization with specific Graphviz options for layout adjustment
dot_exporter = UniqueDotExporter(
    nodes[-1],
    graph="digraph",
    options=["rankdir=TB;", "ranksep=10", "nodesep=0.5"],
    nodenamefunc=lambda node: f'{node.name}',
    nodeattrfunc=lambda node: 'shape=box, fontsize=40, width=2, height=1',
    edgeattrfunc=lambda parent, child: 'arrowhead=none'
)

# Export the tree to a DOT file
dot_exporter.to_dotfile("tree_viz/raptor_tree.dot")
