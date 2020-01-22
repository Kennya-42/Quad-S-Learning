from .cityscapes import Cityscapes
from .imagenet import Imagenet
from .rotloader import Rotloader
from .jigsawLoader import jigsawLoader
from .colorizeLoader import colorizeLoader
from .relpatchLoader import relpatchLoader
from .clusterLoader import clusterLoader

__all__ = [ 'Cityscapes', 'Imagenet', 'Rotloader', 'jigsawLoader', 'colorizeLoader', 'relpatchLoader','clusterLoader']
