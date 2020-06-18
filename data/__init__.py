from .cityscapes import Cityscapes
from .imagenet import Imagenet
from .rotloader import Rotloader
from .jigsawLoader import jigsawLoader
from .colorizeLoader import colorizeLoader
from .relpatchLoader import relpatchLoader
from .clusterLoader import clusterLoader
from .otsuLoader import otsuLoader
from .fulljigsawloader import fulljigsawLoader
from .fulljigsawloader2 import fulljigsawLoader2
from .simclrLoader import simclrloader
from .camvid import Camvid

__all__ = [ 'Cityscapes', 'Imagenet', 'Rotloader', 'jigsawLoader', 'colorizeLoader', 'relpatchLoader','clusterLoader','otsuLoader','fulljigsawLoader','simclrloader','Camvid','fulljigsawLoader2']
