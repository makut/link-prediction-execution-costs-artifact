#!/usr/bin/env python3
"""Read-only source/AST probe; run on targetPython3.10 before any execution."""
from cpu_common_v1 import *

bindings = verify_sources()
proof = read(HERE/'PORTABLE_AST_PINS.json')
rows = {}
for name, entry in proof['files'].items():
    tree = ast.parse((HERE/name).read_text())
    for fn, expected in entry['portable_functions'].items():
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        value = canonical_sha256(node)
        require(value == expected, 'portable AST differs from captured actual target310')
        rows[name+':'+fn] = {'portable_sha256': value, 'matches_actual_target310': True}
print(json.dumps({'status': 'PASS_READONLY_PORTABLE_AST_SOURCE_PROBE', 'python': sys.version,
    'target_python310': sys.version_info[:2] == (3, 10), 'source_manifest_sha256': sha(HERE/'SOURCE_MANIFEST.json'),
    'source_bindings': len(bindings), 'functions': rows, 'output_created': False, 'graph_reads': 0,
    'numerical_imports': 0, 'environment_or_lock_creations': 0, 'scientific_jobs': 0}))
