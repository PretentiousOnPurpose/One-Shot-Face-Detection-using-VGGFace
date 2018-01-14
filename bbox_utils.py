import mxnet as mx
import numpy as np
from mxnet import nd , gluon
from mxnet.contrib.ndarray import MultiBoxPrior
import matplotlib.pyplot as plt

def default_boxes(shape,sizes, ratios):
    return MultiBoxPrior(data=nd.zeros(shape), sizes=sizes, ratios=ratios).asnumpy().reshape((shape[0], shape[2], shape[2], -1, 4))

# def box_to_rect(box, color, width=2):
#     return plt.Rectangle((box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]),fill=False, edgecolor=color, linewidth=width)
#
# def plot_boxes(img, boxes):
#     colors = ['blue', 'green', 'red', 'black', 'magenta']
#     plt.imshow(img)
#     for i in range(len(boxes)):
#         plt.gca().add_patch(box_to_rect(anchors[i,:]*n, colors[i]))
#     plt.show()

