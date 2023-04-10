from typing import Dict, TypedDict, Union, Optional, TypeVar
from anndata import AnnData

AnnDataMapping = Dict[str, AnnData]
T_AnnDataOrDictAnnData = TypeVar("AnnDataOrDictAnnData", AnnData, AnnDataMapping)
