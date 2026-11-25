#!/usr/bin/env python3
"""Whole-member public evidence allowlist for exact recorded E19/E21 assets.

Read-only, bounded streaming inspection. This script never imports an array,
extension, model or producer, and does not contact external infrastructure.
"""
from pathlib import Path, PurePosixPath
import argparse, collections, hashlib, json, re, tarfile
import materialize_pencil_export_v1 as transport

MIB = 1 << 20
CREDS = re.compile(rb'(?i)(?:authorization\s*:\s*bearer\s+|(?:password|oauth_token|yt_token|secret)\s*[:=]\s*[\"\x27])([A-Za-z0-9_\-]{24,})')

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def review(role, archive, extraction_manifest):
    expected = transport.ASSETS[role]
    actual = transport.pin(archive)
    require(actual == {k: expected[k] for k in ('bytes', 'sha256')}, 'exact accepted raw archive required')
    declared = transport.declared_extraction(Path(extraction_manifest).read_bytes())
    seen = {}; counts = collections.Counter(); secret_hits = 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            name = member.name; path = PurePosixPath(name)
            require(str(path) == name and not path.is_absolute() and '..' not in path.parts
                    and '\\' not in name and '\0' not in name, 'canonical public raw member required')
            require(member.isfile() and not member.issparse() and not member.linkname
                    and name in declared and name not in seen, 'declared unique regular raw member required')
            require(not any(part.lower() == 'vk' or part.lower().startswith('vk_') for part in path.parts),
                    'private raw namespace refused')
            require(0 <= member.size <= 16*MIB, 'bounded evidence member required')
            suffix = path.suffix
            if suffix == '.npy':
                require('/real_trace/' in '/'+name or '/tiny_trace/' in '/'+name,
                        'only declared public citation2/tiny replay arrays allowed')
                category = 'public_citation2_or_literal_tiny_trace_and_RNG_state'
            elif suffix == '.so':
                require(role == 'E21' and name == 'cpp_build/native_assoc_v1.so' and member.size == 22352,
                        'only exact recorded own native compilation evidence allowed')
                category = 'own_MIT_native_build_evidence_never_loaded_by_archival_audit'
            else:
                require(suffix in ('.json', '.jsonl', '.log'), 'unexpected raw payload type')
                category = 'genuine_public_science_platform_source_identity_or_lifecycle_metadata'
            h = hashlib.sha256(); size = 0
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(MIB), b''):
                    h.update(block); size += len(block)
                    if suffix != '.npy' and suffix != '.so':
                        secret_hits += len(CREDS.findall(block))
            value = {'bytes': size, 'sha256': h.hexdigest()}
            require(value == declared[name], 'recorded member does not match genuine extraction inventory')
            seen[name] = {**value, 'review_class': category}; counts[category] += 1
    require(len(seen) == expected['members'] and set(seen) == set(declared), 'whole exact raw allowlist required')
    require(secret_hits == 0, 'credential-shaped value requires explicit review')
    return {'kind': 'WHOLE_EXACT_RECORDED_PUBLIC_'+role+'_ALLOWLIST_V1', 'status': 'PASS',
            'role': role, 'archive': actual, 'members': len(seen), 'all_member_hashes_rechecked': True,
            'classification_counts': dict(counts), 'reviewed_payload_allowlist': seen,
            'array_scope': 'Original public ogbl-citation2 and literal tiny traces with RNG states; no full graph, model, checkpoint or private graph/query payload.',
            'source_and_host_provenance': 'Historical author paths, host/process/runtime metadata, root anchors and internal URIs are retained as genuine provenance, never contacted.',
            'native_extension': 'E21 retains exact22352-byte own native build evidence; independent archival auditors read/hash it and never load it.',
            'credential_pattern_hits': secret_hits, 'models_or_extensions_loaded': 0,
            'new_scientific_computation': 0, 'remote_actions': 0, 'YT_writes': 0}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('E19', 'E21'), required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--extraction-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists() and not args.out.is_symlink(), 'fresh public review output required')
    report = review(args.role, args.archive, args.extraction_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps({'status': report['status'], 'role': args.role, 'members': report['members']}))

if __name__ == '__main__':
    main()
