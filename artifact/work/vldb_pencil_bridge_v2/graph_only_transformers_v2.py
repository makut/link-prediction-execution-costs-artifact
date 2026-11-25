"""Explicit optional-vision policy for pinned PENCIL graph-only workers.

No package/source edits, fake operators or changes to graph/model computation.
Transformers4.46.2 imports optional image helpers during BertEncoder import.
An installed incompatible torchvision must not be imported by graph-only jobs.
"""
from pathlib import Path
import hashlib,importlib,importlib.metadata,sys

VERSION='graph_only_transformers_optional_vision_v2'
TRANSFORMERS_VERSION='4.46.2'
IMPORT_UTILS_SHA='9d7f63effbd7c8dcfe69e7f34cae0d164db9cca2d5f4d887405ae0f3e2ba455a'
_RECEIPT=None

def require(ok,message):
    if not ok:raise RuntimeError(message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def configure_graph_only_transformers():
    """Call before any model/vision import, once per fresh worker process."""
    global _RECEIPT
    require(not any(name=='torchvision'or name.startswith('torchvision.')for name in sys.modules),
        'graph-only policy requires a fresh process with no imported torchvision modules')
    require(importlib.metadata.version('transformers')==TRANSFORMERS_VERSION,
        'optional-vision policy pins transformers4.46.2 private availability API')
    utils=importlib.import_module('transformers.utils.import_utils')
    require(digest(utils.__file__)==IMPORT_UTILS_SHA,'pinned transformers import_utils source changed')
    require(type(getattr(utils,'_torchvision_available',None))is bool and callable(getattr(utils,'is_torchvision_available',None)),
        'pinned optional-vision availability API absent')
    if _RECEIPT is not None:
        require(utils._torchvision_available is False and utils.is_torchvision_available()is False,'graph-only optional-vision policy changed after admission')
        return dict(_RECEIPT)
    forbidden=('transformers.image_utils','transformers.modeling_utils','transformers.models.bert.modeling_bert')
    require(not any(name in sys.modules for name in forbidden),'configure graph-only optional dependencies before BERT/image imports')
    before=utils.is_torchvision_available();require(type(before)is bool and before is utils._torchvision_available,'availability API and state disagree')
    try:vision_version=importlib.metadata.version('torchvision')
    except importlib.metadata.PackageNotFoundError:vision_version=None
    # The graph/text BERT path does not consume image transforms or image losses.
    # Leave files, metadata, Torch registration and every upstream file unchanged.
    utils._torchvision_available=False
    require(utils.is_torchvision_available()is False,'failed to disable unused optional vision route')
    _RECEIPT={'version':VERSION,'transformers_version':TRANSFORMERS_VERSION,'import_utils_origin':str(Path(utils.__file__).resolve()),
        'import_utils_sha256':IMPORT_UTILS_SHA,'policy_source_sha256':digest(__file__),
        'installed_torchvision_version':vision_version,'torchvision_available_before':before,'torchvision_available_for_graph_worker':False,
        'torchvision_imported':False,'upstream_source_modified':False,'base_package_modified':False,'fake_operator_registered':False,
        'scope':'Only unused optional vision discovery in pinned Transformers; graph sampler, sparse operator, BERT parameters/forward, precision and input tensors unchanged.'}
    return dict(_RECEIPT)
