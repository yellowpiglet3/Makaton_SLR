import sys
import numpy as np

sys.path.extend(['../'])
from . import tools

num_node = 67
self_link = [(i, i) for i in range(num_node)]
pose_edges = [

    (11,12),

    (11,13),
    (13,15),

    (12,14),
    (14,16),

    (11,23),
    (12,24),

    (23,24),

    (23,25),
    (25,27),

    (24,26),
    (26,28)
]
left_hand = [

(25,26),(26,27),(27,28),(28,29),

(25,30),(30,31),(31,32),(32,33),

(25,34),(34,35),(35,36),(36,37),

(25,38),(38,39),(39,40),(40,41),

(25,42),(42,43),(43,44),(44,45)

]
right_hand = [

(46,47),(47,48),(48,49),(49,50),

(46,51),(51,52),(52,53),(53,54),

(46,55),(55,56),(56,57),(57,58),

(46,59),(59,60),(60,61),(61,62),

(46,63),(63,64),(64,65),(65,66)

]
cross_links = [

(15,25),  # left wrist → left hand root
(16,46)   # right wrist → right hand root

]
# face_neighbor = [
#     (67,68),
#     (68,69),
#     (69,70),

#     (71,72),
#     (72,73),

#     (74,75),
#     (75,76),

#     (77,78),
#     (78,79),

#     (80,81)
# ]

# # for i in range(69):
# #     face_neighbor.append(
# #         (67 + i, 67 + i + 1)
# #    )
# body_face = [
#     (0,67)

# ]
inward = (
    pose_edges
    + left_hand
    + right_hand
    + cross_links
    # + face_neighbor
    # + body_face
)

outward = [(j, i) for (i, j) in inward]

neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

