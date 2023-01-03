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
from .kgflex import KGFlex
from .kgflex_tf import KGFlexTF
from .kgflex_tf2 import KGFlexTF2
from .kgflex_umap import KGFlexUmap

import sys

for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .lightgcn.LightGCN import LightGCN
        from .dgcf.DGCF import DGCF
        from .bprmf.BPRMF import BPRMF
        from .lightgcn_edge import LightGCNEdge
        from .kg_lightgcn_edge import KGLightGCNEdge
        from .kgcn import KGCN
        from .kgat import KGAT
