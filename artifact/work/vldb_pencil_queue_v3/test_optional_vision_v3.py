import copy,hashlib,importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent/'bridge/vldb_pencil_bridge_v2'))
import graph_only_transformers_v2 as g

def setup(monkeypatch,tmp_path,available=True,vision='0.17.2+cu121'):
    source=tmp_path/'import_utils.py';source.write_text('pinned source fixture\n')
    module=SimpleNamespace(__file__=str(source),_torchvision_available=available)
    module.is_torchvision_available=lambda:module._torchvision_available
    monkeypatch.setattr(g,'IMPORT_UTILS_SHA',hashlib.sha256(source.read_bytes()).hexdigest());monkeypatch.setattr(g,'_RECEIPT',None)
    monkeypatch.setattr(g.importlib,'import_module',lambda name:module)
    def version(name):
        if name=='transformers':return '4.46.2'
        if vision is None:raise g.importlib.metadata.PackageNotFoundError(name)
        return vision
    monkeypatch.setattr(g.importlib.metadata,'version',version)
    # Do not mutate unrelated actual imports in the testing process.
    monkeypatch.setattr(g,'sys',SimpleNamespace(modules={}))
    return module,source

def test_incompatible_unused_vision_is_disabled_before_graph_bert_import_and_recorded(monkeypatch,tmp_path):
    module,source=setup(monkeypatch,tmp_path);before=source.read_bytes();result=g.configure_graph_only_transformers()
    assert module._torchvision_available is False and module.is_torchvision_available()is False
    assert result['installed_torchvision_version']=='0.17.2+cu121'and result['torchvision_available_before']is True
    assert source.read_bytes()==before and not result['fake_operator_registered']and not result['base_package_modified']and not result['upstream_source_modified']
    assert g.sys.modules=={}

def test_absent_vision_remains_absent_without_import_or_dependency_creation(monkeypatch,tmp_path):
    module,source=setup(monkeypatch,tmp_path,False,None);result=g.configure_graph_only_transformers()
    assert result['installed_torchvision_version']is None and result['torchvision_available_before']is False and module._torchvision_available is False

def test_idempotent_policy_is_stable_after_graph_model_import(monkeypatch,tmp_path):
    module,source=setup(monkeypatch,tmp_path);first=g.configure_graph_only_transformers();g.sys.modules['transformers.modeling_utils']=object()
    assert g.configure_graph_only_transformers()==first
    module._torchvision_available=True
    with pytest.raises(RuntimeError,match='changed after admission'):g.configure_graph_only_transformers()

@pytest.mark.parametrize('name',['torchvision','torchvision.transforms','transformers.image_utils','transformers.modeling_utils','transformers.models.bert.modeling_bert'])
def test_late_policy_refuses_imported_vision_or_bert_instead_of_silently_repairing(monkeypatch,tmp_path,name):
    module,source=setup(monkeypatch,tmp_path);g.sys.modules[name]=object()
    with pytest.raises(RuntimeError):g.configure_graph_only_transformers()
    assert module._torchvision_available is True

@pytest.mark.parametrize('change',['version','source','api','state'])
def test_unreviewed_transformers_source_version_or_api_refused(monkeypatch,tmp_path,change):
    module,source=setup(monkeypatch,tmp_path)
    if change=='version':monkeypatch.setattr(g.importlib.metadata,'version',lambda name:'5.0')
    elif change=='source':source.write_text('changed source')
    elif change=='api':module._torchvision_available=1
    else:module.is_torchvision_available=lambda:False
    with pytest.raises(RuntimeError):g.configure_graph_only_transformers()

def test_pinned_wheel_source_has_exact_reviewed_boolean_gate_and_no_vision_import_in_bert_source():
    import ast,zipfile
    wheel=Path(__file__).resolve().parents[1]/'pencil_bridge_wheelhouse_linux_cp310_matched/transformers-4.46.2-py3-none-any.whl'
    if not wheel.exists():pytest.skip('optional matched offline wheel fixture is not included in small source packet')
    with zipfile.ZipFile(wheel)as archive:
        data=archive.read('transformers/utils/import_utils.py');assert hashlib.sha256(data).hexdigest()==g.IMPORT_UTILS_SHA
        tree=ast.parse(data);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='is_torchvision_available')
        assert len(fn.body)==1 and isinstance(fn.body[0],ast.Return)and isinstance(fn.body[0].value,ast.Name)and fn.body[0].value.id=='_torchvision_available'
        image=ast.parse(archive.read('transformers/image_utils.py'));guard=next(n for n in ast.walk(image)if isinstance(n,ast.If)and
            isinstance(n.test,ast.Call)and isinstance(n.test.func,ast.Name)and n.test.func.id=='is_torchvision_available')
        assert any(isinstance(n,ast.ImportFrom)and n.module=='torchvision.transforms'for n in guard.body)
        bert=ast.parse(archive.read('transformers/models/bert/modeling_bert.py'))
        assert not any(isinstance(n,ast.ImportFrom)and(n.module or'').startswith('torchvision')for n in ast.walk(bert))
