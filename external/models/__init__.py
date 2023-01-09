def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


from .most_popular import MostPop
from .Proxy import ProxyRecommender

from .ktup import KTUP
from .cke import CKE
from .cofm import CoFM

import sys

for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        from .kgflex import KGFlex
    elif _backend == "pytorch":
        from .lightgcn.LightGCN import LightGCN
        from .dgcf.DGCF import DGCF
        from .bprmf.BPRMF import BPRMF
        from .lightgcn_edge import LightGCNEdge
        from .kgtore.KGTORE import KGTORE
        from .kgcn import KGCN
        from .kgat import KGAT
