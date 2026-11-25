"""Compatibility-only regression against actual captured Python3.10 AST pins."""
from pathlib import Path
import ast, copy, hashlib, json, subprocess, sys
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import portable_ast_v1 as portable


def project310(node):
    node = copy.deepcopy(node)
    for value in ast.walk(node):
        if 'type_params' in value._fields:
            assert value.type_params == []
            value._fields = tuple(name for name in value._fields if name != 'type_params')
            del value.type_params
    return node


def test_all7_frozen_functions_match_actual_target310_and_preserved312_lineage():
    pins = json.loads((HERE/'PORTABLE_AST_PINS.json').read_text())
    raw = (HERE/'V1_TARGET310_PORTABILITY_DIAGNOSIS.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pins['actual_target310_diagnosis_sha256']
    diagnosis = json.loads(raw); count = 0
    for name, entry in pins['files'].items():
        body = (HERE/name).read_bytes()
        assert {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()} == entry['extracted_file_identity']
        tree = ast.parse(body.decode())
        for fn, expected in entry['portable_functions'].items():
            node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == fn)
            observed = diagnosis['functions'][name+':'+fn]
            assert observed['target_native_AST_sha256'] == expected == portable.canonical_sha256(node)
            assert portable.canonical_sha256(project310(node)) == expected
            assert entry['unchanged_original_lineage']['functions'][fn] == observed['frozen_local312_AST_sha256']
            assert observed['adding_only_empty_type_params_matches_frozen'] is True
            count += 1
    assert count == 7


@pytest.mark.parametrize('source', ['def f():\n pass', 'async def f():\n pass', 'class C:\n pass'])
def test_only_empty_type_params_field_is_a_crossruntime_equivalence(source):
    node = ast.parse(source).body[0]; old = project310(node)
    simulated312 = copy.deepcopy(old)
    simulated312._fields = (*simulated312._fields, 'type_params'); simulated312.type_params = []
    assert portable.canonical_dump(old) == portable.canonical_dump(simulated312)
    assert not hasattr(old, 'type_params') or 'type_params' not in old._fields


@pytest.mark.parametrize('value', [[ast.Name(id='T', ctx=ast.Load())], None, ()])
def test_nonempty_or_nonlist_type_parameters_cannot_be_normalized(value):
    node = ast.parse('def f():\n pass').body[0]
    if 'type_params' not in node._fields: node._fields = (*node._fields, 'type_params')
    node.type_params = value
    with pytest.raises(ValueError, match='only explicitly empty'): portable.canonical_dump(node)


def test_type_params_on_unsupported_AST_node_is_rejected():
    node = ast.Name(id='x', ctx=ast.Load()); node._fields = (*node._fields, 'type_params'); node.type_params = []
    with pytest.raises(ValueError, match='unsupported'): portable.canonical_dump(node)


@pytest.mark.parametrize('change', ['body', 'args', 'decorator', 'returns', 'annotation', 'default', 'type_comment'])
def test_all_semantic_fields_remain_part_of_identity(change):
    original = ast.parse('def f(a: int = 1) -> int:\n return a').body[0]
    node = copy.deepcopy(original)
    if change == 'body': node.body = [ast.Return(value=ast.Constant(value=2))]
    elif change == 'args': node.args.args[0].arg = 'b'
    elif change == 'decorator': node.decorator_list = [ast.Name(id='decorator', ctx=ast.Load())]
    elif change == 'returns': node.returns = ast.Name(id='str', ctx=ast.Load())
    elif change == 'annotation': node.args.args[0].annotation = ast.Name(id='str', ctx=ast.Load())
    elif change == 'default': node.args.defaults[0] = ast.Constant(value=2)
    else: node.type_comment = '(int) -> int'
    assert portable.canonical_sha256(node) != portable.canonical_sha256(original)


def test_available_python313_serialization_matches_all_actual310_pins():
    alternate = Path(sys.executable).resolve().parents[2]/'3.13.3/bin/python3'
    if not alternate.is_file(): pytest.skip('optional independent local Python3.13 interpreter unavailable')
    code = "from pathlib import Path;import ast,json,sys;sys.path.insert(0,sys.argv[1]);from portable_ast_v1 import canonical_sha256;p=Path(sys.argv[1]);a=json.loads((p/'PORTABLE_AST_PINS.json').read_text());n=0\nfor name,e in a['files'].items():\n t=ast.parse((p/name).read_text())\n for fn,h in e['portable_functions'].items():\n  x=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==fn);assert canonical_sha256(x)==h;n+=1\nassert n==7;print('7 actual target310 pins matched by Python3.13')"
    result = subprocess.run([str(alternate), '-B', '-c', code, str(HERE)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '7 actual target310 pins matched by Python3.13'
