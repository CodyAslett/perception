class Node():
    def __init__(self, name = "", children = {}):
        self.name = name
        self.children = children

    #returns if the children is empty
    def isLeaf(self):
        return not self.children

    #appends the child to the children dictionary
    def appendChild(self, attribute, value):
        self.children[attribute] = value