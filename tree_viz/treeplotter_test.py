from treeplotter.tree import Node, Tree
from treeplotter.plotter import create_tree_diagram
root = Node(value=1.0, name=None)

child1 = Node(value=0.5, name=None)
child2 = Node(value=1.0, name=None)
child3 = Node(value=3.0, name="A")
root.children = {child1, child2, child3}
tree = Tree(root=root)

create_tree_diagram(
    tree=tree,
    save_path="test_tree_image",
    webshot=True,
    verbose=True
)
