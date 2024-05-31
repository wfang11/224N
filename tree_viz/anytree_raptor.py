from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter
import json

# Load the tree data from a JSON file
with open("trees/GPT_1000_node_data.json", "r") as f:
    data = json.loads(f.read())

## filter out layer 0 and 1 and 2 ?
filtered_data = {}
for key, value in data.items():
    if int(value['Layer']) > 3: 
        filtered_data[key] = value

# Determine the maximum layer among all nodes to help define the root node
MAX_LAYER = max(int(node['Layer']) for node in data.values())

# Identify nodes that are located at the maximum layer to set them as children of the new root node
max_layer_nodes = [int(key) for key, value in data.items() if value['Layer'] == MAX_LAYER]

# Add a new root node with children that are the nodes from the maximum layer
data['-1'] = {
    'Node': -1,
    'Layer': MAX_LAYER + 1,
    'Text': 'ROOT',
    'Children': max_layer_nodes
}

# Initialize nodes without parents from the data
nodes = {int(key): Node(name=data[key]['Text']) for key in data}

# Establish parent-child relationships based on the 'Children' list in the data
for key, value in data.items():
    for child in value['Children']:
        nodes[child].parent = nodes[int(key)]

# Configure the tree visualization with specific Graphviz options for layout adjustment
dot_exporter = UniqueDotExporter(
    nodes[-1],
    graph="digraph",
    options=["rankdir=TB;", "ranksep=10", "nodesep=0.5"],  # Adjusted spacing for better vertical layout
    nodenamefunc=lambda node: f'{node.name}',
    nodeattrfunc=lambda node: 'shape=box, fontsize=40, width=2, height=1',  # Increased node size
    edgeattrfunc=lambda parent, child: 'arrowhead=none'
)

# Export the tree to a DOT file and then render to an image
dot_exporter.to_dotfile("tree_viz/raptor_mini.dot")
# dot_exporter.to_picture("tree_viz/raptor_mini_large_nodes.png")
