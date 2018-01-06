from ssd_model import ssd, class_pred, box_pred
from bbox_utils import default_boxes, box_to_rect, plot_boxes

SSD = ssd(300)
SSD.summary()
