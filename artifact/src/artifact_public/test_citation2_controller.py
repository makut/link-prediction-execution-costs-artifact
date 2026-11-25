"""Tiny end-to-end orchestration fixtures; no network or dataset downloads."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_public import reproduce_citation2 as controller


def fixture(tmp_path, monkeypatch):
    code = tmp_path / 'code'; code.mkdir()
    hashes = {}
    for name in ('experiment_configs/seal_revision_2026-11/prepare_citation2.py',
                 'scripts/prepare_ogb_dataset.py', 'scripts/transpose_graph_csr.py',
                 'scripts/convert_graph_csr_to_upper.py'):
        path = code / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen source\n')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    files = [dict(path=f'array{i}.bin', bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())
             for i in range(19)]
    expected = tmp_path / 'expected.json'
    expected.write_text(json.dumps({'files': files, 'preparer_source_sha256': hashes,
                                    'recorded_archive_sha256': '0' * 64}))
    item = {'name': 'citation-v2.zip', 'bytes': 11,
            'url': 'https://snap.stanford.edu/ogb/data/linkproppred/citation-v2.zip'}
    archive = tmp_path / item['name']; archive.write_bytes(b'officialzip')
    receipt = tmp_path / 'official_receipt.json'
    receipt.write_text(json.dumps({**item, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                                   'source_kind': 'fresh_official_public_download'}))
    monkeypatch.setitem(controller.SOURCES, 'citation2', [item])
    monkeypatch.setattr(controller, 'memory_available', lambda: 64 << 30)
    monkeypatch.setattr(controller.shutil, 'disk_usage', lambda path: SimpleNamespace(free=128 << 30))
    def forbidden(*args, **kwargs):
        raise AssertionError('provisioned archive mode must not download')
    monkeypatch.setattr(controller, 'download', forbidden)
    args = ['--work-parent', str(tmp_path / 'runs'), '--code-root', str(code),
            '--expected', str(expected), '--archive-existing', str(archive),
            '--official-download-receipt', str(receipt), '--execute']
    return code, files, receipt, archive, args


def test_provisioned_official_archive_runs_unchanged_prep_and_all_19_checks(tmp_path, monkeypatch):
    code, files, receipt, archive, args = fixture(tmp_path, monkeypatch)
    def run(argv, **kwargs):
        assert Path(argv[1]) == code / 'experiment_configs/seal_revision_2026-11/prepare_citation2.py'
        assert Path(argv[argv.index('--zip') + 1]) == archive
        assert Path(argv[argv.index('--source-manifest') + 1]) == receipt
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        prepared = Path(argv[argv.index('--root') + 1]) / 'prepared'; prepared.mkdir(parents=True)
        for entry in files:
            (prepared / entry['path']).write_bytes(b'abc')
    monkeypatch.setattr(controller.subprocess, 'run', run)
    controller.main(args)
    runs = list((tmp_path / 'runs').iterdir()); assert len(runs) == 1
    result = json.loads((runs[0] / 'PUBLIC_19_PAYLOAD_IDENTITY.json').read_text())
    assert result['payload_count'] == 19 and result['full_public_reproduction_verified']
    assert archive.read_bytes() == b'officialzip'


def test_corrupt_provisioned_archive_fails_and_preserves_evidence(tmp_path, monkeypatch):
    _, _, _, archive, args = fixture(tmp_path, monkeypatch)
    archive.write_bytes(b'corruptzip!')
    with pytest.raises(ValueError, match='integrity'):
        controller.main(args)
    run = next((tmp_path / 'runs').iterdir())
    status = json.loads((run / 'STATUS.json').read_text())
    assert status['stage'] == 'failed' and not status['full_public_reproduction_verified']
    assert not (run / 'PUBLIC_19_PAYLOAD_IDENTITY.json').exists()


def test_changed_preparer_fails_before_any_run_or_download(tmp_path, monkeypatch):
    code, _, _, _, args = fixture(tmp_path, monkeypatch)
    (code / 'scripts/prepare_ogb_dataset.py').write_text('changed\n')
    with pytest.raises(ValueError, match='source differs'):
        controller.main(args)
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('budget', ['nan', 'inf', '-inf'])
def test_nonfinite_ram_allowances_rejected_before_work(tmp_path, budget):
    with pytest.raises(SystemExit):
        controller.main(['--work-parent', str(tmp_path / 'runs'), '--memory-budget-gib=' + budget])
    assert not (tmp_path / 'runs').exists()
