from .UNet import UNet
from .SwinUnet import SwinUnet
from .TransUnet import TransUnet
from .SwinU_TRM import SwinU_TRM

model_dict = {
    "UNet": UNet,
    "SwinUnet": SwinUnet,
    "TransUnet": TransUnet,
    "SwinU_TRM": SwinU_TRM,
}
