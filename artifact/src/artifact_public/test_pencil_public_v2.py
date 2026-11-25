"""Portable protocol, provenance, refusal and actual pinned-BERT regressions.

No real graph, CUDA, native torch-sparse, training or full48-job run is claimed.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from . import systems_common as c
from . import reproduce_pencil_v2 as p
from . import pencil_environment_v2 as env


def arguments(tmp_path):
    return argparse.Namespace(out=tmp_path / 'out', execute=False,
        citation2_root=tmp_path / 'public_c2', validation_root=None,
        coordination_dir=None, memory_budget_gib=180., max_output_gib=2., max_hours=12.)


def normalize(tokens):
    return ['<path>' if str(x).startswith('/') else x for x in tokens]


def test_public48_job_protocol_preserves_every_scientific_argument_except_optional_private_plan(tmp_path):
    args = arguments(tmp_path)
    public = p.make_plan(args)
    source = c.ROOT / 'metadata/protocols/pencil_original_scientific_builder.py'
    namespace = {'Path': Path, 'BRIDGE': p.BRIDGE, 'ARMS': p.ARMS, 'REPLAY_ARMS': p.REPLAY_ARMS,
        'layout': lambda _: {'out': args.out,
            'cpp_module': p.BRIDGE / 'corrected_native/native_adapter_v1.py',
            'cpp_library': args.out / 'cpp_build/native_assoc_v1.so',
            'cpp_receipt': args.out / 'cpp_build/build_receipt_v1.json'}}
    exec(compile(source.read_text(), str(source), 'exec'), namespace)
    frozen = namespace['make_jobs'](tmp_path / 'unused', sys.executable)
    assert len(public['jobs']) == len(frozen) == 48
    assert sum(row['kind'] == 'benchmark' for row in public['jobs']) == 36
    for current, original in zip(public['jobs'], frozen):
        assert {k: v for k, v in current.items() if k not in ('argv', 'output')} == {
            k: v for k, v in original.items() if k not in ('argv', 'output')}
        command = list(original['argv'])
        if '--quality-plan' in command:
            index = command.index('--quality-plan')
            del command[index:index + 2]
        assert normalize(current['argv']) == normalize(command)
        assert '--quality-plan' not in current['argv']
        assert not any('/home/jovyan/' in str(v) for v in current['argv'])
    assert [j['name'] for j in public['setup_jobs']] == [
        'environment_before', 'prepare_public_validation', 'build_public_cpp']
    assert public['finalization_jobs'][0]['name'] == 'environment_after'
    assert not args.out.exists()


def test_offline_cli_prints_plan_without_importing_numerical_modules_or_writing_outputs(tmp_path):
    args = arguments(tmp_path)
    code = """import json,sys
from artifact_public.reproduce_pencil_v2 import main
assert main(sys.argv[1:])==0
assert not any(n in sys.modules for n in ('numpy','torch','transformers','torch_sparse'))
"""
    result = subprocess.run([sys.executable, '-B', '-c', code,
        '--citation2-root', str(args.citation2_root), '--out', str(args.out)],
        env={**os.environ, 'PYTHONPATH': str(c.CODE)}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['execute'] is False and plan['repaired_frozen_sources']['files'] == 82
    assert not args.out.exists() and not args.citation2_root.exists()


def test_exact_v3_validator_function_bodies_are_unchanged():
    receipt = c.read(c.ROOT / 'metadata/PENCIL_V2_DERIVATION.json')
    current = ast.parse((c.CODE / 'artifact_public/pencil_validation_v2.py').read_text())
    assert receipt['original_queue_source_sha256'] == 'dda406371cfe9a7d5de8415224930959e970f2b9c948f56d37dca09fffaf68cd'
    for name in ('validate_job', 'crossjob_scores'):
        node = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name)
        actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        assert actual == receipt['scientific_validation_ast_sha256'][name]


def test_changed_frozen_source_and_rebound_manifest_are_refused(tmp_path, monkeypatch):
    old = c.ROOT
    definition = c.read(env.FROZEN)
    for name in definition['files']:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old / name, target)
    path = tmp_path / 'metadata/PENCIL_V2_FROZEN_SOURCES.json'
    path.write_bytes(env.FROZEN.read_bytes())
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(env, 'FROZEN', path)
    assert env.verify_candidate_sources()['files'] == 82
    target = tmp_path / 'work/vldb_pencil_bridge_v2/graph_only_transformers_v2.py'
    target.write_bytes(target.read_bytes() + b'\n')
    with pytest.raises(RuntimeError, match='source differs'):
        env.verify_candidate_sources()
    item = definition['files']['work/vldb_pencil_bridge_v2/graph_only_transformers_v2.py']
    item.update(bytes=target.stat().st_size, sha256=c.sha(target))
    path.write_text(json.dumps(definition))
    with pytest.raises(RuntimeError, match='manifest differs'):
        env.verify_candidate_sources()


def test_public_19_declarations_and_used_payload_verification_need_no_private_plan(tmp_path, monkeypatch):
    expected = c.read(c.ROOT / 'metadata/citation2_prepared_expected.json')
    manifest = {'variant': 'strict_clean', 'files': expected['files']}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    from . import verify_payloads
    observed = []
    monkeypatch.setattr(verify_payloads, 'verify', lambda root, definition: observed.append(definition) or {})
    result = p.verify_public_inputs(tmp_path)
    assert result['strict_manifest_19_payload_declarations_match'] is True
    assert result['all19_payload_bytes_independently_rehashed_here'] is False
    assert {r['path'] for r in observed[0]['files']} == {
        'train_csr/edge_starts.bin', 'train_csr/edge_ends.bin', 'node_feat.npy', 'valid_edge.npy', 'valid_edge_neg.npy'}
    manifest['files'][0]['sha256'] = '0' * 64
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match='exact19'):
        p.verify_public_inputs(tmp_path)


@pytest.mark.parametrize('mutation', ['policy', 'library', 'wrapper', 'numpy'])
def test_drift_between_environment_admission_and_driver_is_refused(mutation):
    admitted = {'runtime_compatibility': {'pinned': True}, 'torch': '2.5.0+cu121',
        'torch_sparse_version': '0.6.18', 'pyg': '2.7.0',
        'saint_wrapper': {'sha256': 'wrapper', 'git_blob': 'blob'},
        'torch_sparse_binaries': {'native.so': {'sha256': 'lib'}},
        'packages': {k: 'version' for k in ('numpy', 'numba', 'scipy', 'transformers', 'wandb')}}
    native = {'torch': admitted['torch'], 'torch_sparse': admitted['torch_sparse_version'],
        'pyg': admitted['pyg'], 'wrapper_sha256': 'wrapper', 'wrapper_git_blob': 'blob',
        'extension_binaries': [{'name': 'native.so', 'sha256': 'lib'}], 'software_versions': dict(admitted['packages'])}
    raw = {'runtime_compatibility': dict(admitted['runtime_compatibility']),
        'audit_environment': {'runtime_compatibility': dict(admitted['runtime_compatibility'])}, 'native_setup': native}
    assert env.validate_driver_environment(raw, admitted)
    if mutation == 'policy': raw['runtime_compatibility']['pinned'] = False
    if mutation == 'library': native['extension_binaries'][0]['sha256'] = 'changed'
    if mutation == 'wrapper': native['wrapper_git_blob'] = 'changed'
    if mutation == 'numpy': native['software_versions']['numpy'] = 'changed'
    with pytest.raises(RuntimeError):
        env.validate_driver_environment(raw, admitted)


def test_existing_output_and_source_output_are_refused_before_execution(tmp_path):
    target = tmp_path / 'existing'
    target.mkdir()
    sentinel = target / 'preserve'
    sentinel.write_text('unchanged')
    with pytest.raises(RuntimeError, match='fresh output'):
        p.main(['--citation2-root', str(tmp_path / 'c2'), '--out', str(target)])
    assert sentinel.read_text() == 'unchanged'
    with pytest.raises(RuntimeError, match='outside'):
        p.main(['--citation2-root', str(tmp_path / 'c2'), '--out', str(c.CODE / 'refused')])


def test_actual_pinned_bert_forward_survives_poisoned_optional_vision(tmp_path):
    pytest.importorskip('transformers')
    pytest.importorskip('torch')
    package = tmp_path / 'torchvision'
    package.mkdir()
    (package / '__init__.py').write_text("raise RuntimeError('operator torchvision::nms does not exist')\n")
    dist = tmp_path / 'torchvision-0.17.2+cu121.dist-info'
    dist.mkdir()
    (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: torchvision\nVersion: 0.17.2+cu121\n')
    variables = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(tmp_path), str(p.BRIDGE))),
        'HF_HUB_OFFLINE': '1', 'WANDB_MODE': 'disabled', 'PYTHONDONTWRITEBYTECODE': '1',
        'HF_HOME': str(tmp_path / 'hf'), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1'}
    baseline = subprocess.run([sys.executable, '-B', '-c',
        'from transformers.models.bert.modeling_bert import BertEncoder'],
        env=variables, capture_output=True, text=True, timeout=60)
    assert baseline.returncode != 0 and 'operator torchvision::nms does not exist' in baseline.stderr
    code = """import json,sys,torch
from graph_only_transformers_v2 import configure_graph_only_transformers
receipt=configure_graph_only_transformers()
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
cfg=BertConfig(hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64)
cfg._attn_implementation='sdpa'
torch.manual_seed(42)
model=BertEncoder(cfg).eval()
with torch.no_grad(): result=model(torch.randn(2,7,32),attention_mask=torch.zeros(2,1,7,7)).last_hidden_state
assert list(result.shape)==[2,7,32] and torch.isfinite(result).all()
assert not any(n=='torchvision' or n.startswith('torchvision.') for n in sys.modules)
print(json.dumps({'shape':list(result.shape),'receipt':receipt,'torch':torch.__version__}))
"""
    fixed = subprocess.run([sys.executable, '-B', '-c', code], env=variables,
        capture_output=True, text=True, timeout=60)
    assert fixed.returncode == 0, fixed.stderr
    actual = json.loads(fixed.stdout)
    assert actual['shape'] == [2, 7, 32]
    assert actual['receipt']['fake_operator_registered'] is False
