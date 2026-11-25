"""Small adversarial fixtures for the public acquisition and conversion path."""
from pathlib import Path
import hashlib
import io
import json
import struct
import tarfile
import zlib

import numpy as np
import pytest

from artifact_public.verify_payloads import verify
from artifact_public.prepare_friendster import extract_evaluation
from artifact_public import prepare_friendster as friendster_prepare
from artifact_public.prepare_friendster_full import main as full_main
from artifact_public import download as public_download
from scripts.prepare_friendster_csr import build_csr, iter_zlib_binary_chunks, load_pairs_u64le, load_vertices_u64le


def test_numeric_identity_rejects_corruption_and_path_traversal(tmp_path):
    (tmp_path / 'array.bin').write_bytes(b'abc')
    entry = {'path': 'array.bin', 'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}
    assert verify(tmp_path, {'files': [entry]})['payload_count'] == 1
    (tmp_path / 'array.bin').write_bytes(b'bad')
    with pytest.raises(ValueError, match='SHA256'):
        verify(tmp_path, {'files': [entry]})
    with pytest.raises(ValueError, match='unsafe'):
        verify(tmp_path, {'files': [{**entry, 'path': '../array.bin'}]})


def test_real_binary_parser_and_loop_duplicate_csr_semantics(tmp_path):
    pairs = np.array([[0, 1], [0, 1], [1, 1], [2, 3]], dtype='<u8')
    edges = tmp_path / 'edges.ez'; edges.write_bytes(zlib.compress(pairs.tobytes()))
    vertices = tmp_path / 'vertices.vz'
    vertices.write_bytes(zlib.compress(np.arange(5, dtype='<u8').tobytes()))
    chunks = list(iter_zlib_binary_chunks(edges, 16, chunk_bytes=16, read_bytes=3))
    assert b''.join(chunks) == pairs.tobytes()
    src, dst = load_pairs_u64le(edges, expected_edges=4, chunk_bytes=32)
    assert np.array_equal(load_vertices_u64le(vertices, 16), np.arange(5))
    starts, ends, stats = build_csr(src, dst, 5)
    assert np.array_equal(starts, [0, 1, 2, 3, 4])
    assert np.array_equal(ends, [1, 0, 3, 2])
    assert stats['self_loops_dropped'] == 1
    assert stats['duplicate_entries_dropped'] == 2
    assert stats['isolated_nodes'] == 1
    assert starts.astype('<i8').tobytes() == struct.pack('<5q', 0, 1, 2, 3, 4)


def test_truncated_zlib_binary_record_rejected(tmp_path):
    path = tmp_path / 'bad.ez'; path.write_bytes(zlib.compress(b'123'))
    with pytest.raises(ValueError, match='record size'):
        list(iter_zlib_binary_chunks(path, 16, chunk_bytes=16))


def test_public_bundle_exact_allowlist_rejects_extra_member(tmp_path):
    bundle = tmp_path / 'extra.tgz'
    with tarfile.open(bundle, 'w:gz') as archive:
        info = tarfile.TarInfo('../escape'); info.size = 1
        archive.addfile(info, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='allowlist'):
        extract_evaluation(bundle, tmp_path / 'out', {'evaluation_files': {}})
    assert not (tmp_path / 'out').exists()


def test_full_converter_refuses_existing_before_loading_inputs(tmp_path):
    out = tmp_path / 'existing'; out.mkdir()
    marker = out / 'keep'; marker.write_text('keep')
    with pytest.raises(FileExistsError):
        full_main(['--inputs', str(tmp_path / 'missing'), '--out', str(out), '--execute'])
    assert marker.read_text() == 'keep'


def response(body):
    class Response(io.BytesIO):
        headers = {}
    return Response(body)


def test_public_download_verifies_checksum_and_refuses_changed_existing(tmp_path, monkeypatch):
    item = {'name': 'tiny', 'bytes': 3, 'url': 'https://example.invalid/tiny',
            'md5': hashlib.md5(b'abc').hexdigest()}
    monkeypatch.setattr(public_download.urllib.request, 'urlopen', lambda *a, **kw: response(b'abc'))
    target = tmp_path / 'tiny'
    first = public_download.download(item, target)
    assert first['sha256'] == hashlib.sha256(b'abc').hexdigest()
    assert public_download.download(item, target) == first
    target.write_bytes(b'bad')
    with pytest.raises(ValueError, match='identity'):
        public_download.download(item, target)
    assert target.read_bytes() == b'bad'


def test_public_download_stops_at_pinned_size_without_publishing_file(tmp_path, monkeypatch):
    item = {'name': 'tiny', 'bytes': 3, 'url': 'https://example.invalid/tiny'}
    monkeypatch.setattr(public_download.urllib.request, 'urlopen', lambda *a, **kw: response(b'abcdef'))
    target = tmp_path / 'tiny'
    with pytest.raises(ValueError, match='exceeded'):
        public_download.download(item, target)
    assert not target.exists() and not target.with_name('tiny.download.json').exists()


def test_exact_allowlist_still_rejects_traversal_path(tmp_path):
    bundle = tmp_path / 'bad.tgz'
    with tarfile.open(bundle, 'w:gz') as archive:
        info = tarfile.TarInfo('../escape'); info.size = 1
        archive.addfile(info, io.BytesIO(b'x'))
    expected = {'evaluation_files': {'../escape': {'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}}}
    with pytest.raises(ValueError, match='unsafe.*path'):
        extract_evaluation(bundle, tmp_path / 'out', expected)
    assert not (tmp_path / 'escape').exists()


def test_friendster_split_memory_preflight_precedes_graph_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(friendster_prepare, 'memory_available', lambda: 0)
    def forbidden(*args, **kwargs):
        raise AssertionError('graph must not be read after failed RAM preflight')
    monkeypatch.setattr(friendster_prepare, 'verify', forbidden)
    with pytest.raises(RuntimeError, match='available RAM'):
        friendster_prepare.main(['--full-graph', str(tmp_path / 'absent'),
                                 '--out', str(tmp_path / 'out'), '--threads', '1'])
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('budget', ['nan', 'inf', '-inf'])
def test_nonfinite_download_budgets_rejected_before_network(tmp_path, budget):
    with pytest.raises(SystemExit):
        public_download.main(['citation2', '--out', str(tmp_path / 'out'), '--max-download-gib=' + budget])
    assert not (tmp_path / 'out').exists()
