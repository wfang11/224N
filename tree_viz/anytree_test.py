from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter
udo = Node("Udo")
marc = Node("Marc", parent=udo)
lian = Node("Lian", parent=marc)
dan = Node("Dan", parent=udo)
jet = Node("Jet", parent=dan)
jan = Node("Jan", parent=dan)
joe = Node("Joe", parent=dan)
for pre, fill, node in RenderTree(udo):
    print("%s%s" % (pre, node.name))


# graphviz needs to be installed for the next line!
UniqueDotExporter(udo).to_picture("tree_viz/udo.png")