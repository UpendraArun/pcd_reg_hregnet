from .chamfer_loss import ChamferDistanceLoss
from .losses import transformation_loss, prob_chamfer_loss, matching_loss
from mi_loss_v2 import LocalinfolossNet, GlobalinfolossNet, DeepMILoss
#from mi_loss import LocalinfolossNet, GlobalinfolossNet, DeepMILoss
from .overlap_circle_loss import OverlapAwareCircleLoss