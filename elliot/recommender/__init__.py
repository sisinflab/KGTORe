"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .base_recommender_model import BaseRecommenderModel

from .unpersonalized import Random, MostPop
from .knowledge_aware import KaHFM, KaHFMBatch, KaHFMEmbeddings, KGIN
from .knn import ItemKNN, UserKNN, AttributeItemKNN, AttributeUserKNN
from .generic import ProxyRecommender
from .autoencoders import EASER

