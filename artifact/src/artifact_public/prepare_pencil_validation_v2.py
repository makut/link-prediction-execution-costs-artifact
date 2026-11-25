"""Regenerate only512 frozen Citation2 validation groups; never open test arrays."""
import argparse
import json
from pathlib import Path

from . import systems_common as c


def prepare(prepared, out):
    helper = c.load_file(c.CODE / 'experiment_configs/seal_revision_2026-11/quality/prepare_sources.py', 'public_frozen_subset_preparer')
    return helper.freeze_source_subset(pos_path=prepared / 'valid_edge.npy', neg_path=prepared / 'valid_edge_neg.npy',
        output_dir=out, name='valid', count=512, seed=20261124, expected_negatives=1000, chunk_rows=250_000)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--citation2-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    prepared, out = args.citation2_root.resolve(), args.out.resolve()
    c.require(not out.exists() and not out.is_relative_to(prepared) and not prepared.is_relative_to(out)
              and not out.is_relative_to(c.ROOT), 'fresh validation output outside source/input required')
    c.frozen_sources()
    from .pencil_environment_v2 import verify_candidate_sources
    verify_candidate_sources()
    if not args.execute:
        print(json.dumps({'execute': False, 'split': 'valid', 'sources': 512, 'seed': 20261124, 'test_payload_access': False}))
        return 0
    result = prepare(prepared, out)
    contract = c.read(c.ROOT / 'work/vldb_pencil_bridge_v2/input_contract.json')['payloads']
    for key, expected in (('positive', 'valid_positive'), ('negative', 'valid_negative')):
        c.require(result['outputs'][key]['sha256'] == contract[expected]['sha256'], 'exact original validation subset hash differs')
    print(json.dumps({'status': 'complete', 'validation_sources': 512, 'test_payload_access': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
