from collections import OrderedDict

import numba as nb
from numba import deferred_type, float32, int8, jit, optional, types
from numba.experimental import jitclass

node_type = deferred_type()


LEAF = 0
VALUE = 1
EVENT = 2


@jitclass()
class Node:
    type: int  # 0 = leaf, 1 = value, 2 = event
    y: optional(int8)
    v: optional(float32)
    l: optional(types.unicode_type)
    d: optional(float32)
    true_child: optional(node_type)
    false_child: optional(node_type)

    def __init__(self, type=LEAF, y=None, v=None, l=None, d=None):
        self.true_child = None
        self.false_child = None
        self.type = type
        self.y = y
        self.v = v
        self.l = l
        self.d = d


node_type.define(Node.class_type.instance_type)
# Create node and assign children works:
Node().true_child = Node()

print("Node class:", Node.class_type.instance_type)
