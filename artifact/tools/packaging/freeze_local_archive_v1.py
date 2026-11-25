#!/usr/bin/env python3
"""Create and fully read back a fresh local regular-member code archive.

No publication, git, external connection, graph or model access. The external
receipt binds the archive hash without creating a self-referential manifest.
"""
from pathlib import Path, PurePosixPath
import argparse, gzip, hashlib, json, os, shutil, tarfile

MIB = 1 << 20
RESERVE = 3 * (1 << 29)
MAX_CODE = 384 * MIB
PREFIX = 'lpp_vldb_artifact_v0.5.0/'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular local file required')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}

def safe(name):
    p = PurePosixPath(name)
    require(name and str(p) == name and not p.is_absolute() and '..' not in p.parts
            and '\\' not in name and '\0' not in name, 'canonical member path required')
    return name

def verify_snapshot(root):
    root = Path(root).resolve()
    require((root/'MANIFEST.json').stat().st_size < 4*MIB, 'bounded snapshot manifest required')
    manifest = json.loads((root/'MANIFEST.json').read_text())
    require(manifest['version'] == '0.5.0' and manifest['manifest_excludes_itself'] is True,
            'stable v0.5.0 snapshot manifest required')
    require(manifest['file_count'] == len(manifest['files']), 'exact snapshot file count required')
    rows = {safe(name): {key: row[key] for key in ('bytes', 'sha256')}
            for name, row in manifest['files'].items()}
    require('MANIFEST.json' not in rows, 'manifest must exclude itself')
    rows['MANIFEST.json'] = pin(root/'MANIFEST.json')
    observed = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'source symlink refused')
        if path.is_dir():
            continue
        require(path.is_file(), 'source special file refused')
        observed.add(path.relative_to(root).as_posix())
    require(observed == set(rows), 'snapshot disk surface differs from manifest')
    require(sum(row['bytes'] for row in rows.values()) < MAX_CODE, 'bounded384MiB code snapshot required')
    for name, row in rows.items():
        require(pin(root/name) == row, 'snapshot bytes changed: '+name)
    return rows

def readback(archive, rows):
    seen = set()
    with gzip.open(archive, 'rb') as decompressed, tarfile.open(fileobj=decompressed, mode='r|') as tar:
        for member in tar:
            require(member.name.startswith(PREFIX), 'exact archive prefix required')
            name = safe(member.name[len(PREFIX):])
            require(name in rows and name not in seen and member.isfile() and not member.issparse()
                    and not member.linkname, 'unique regular archive member required')
            row = rows[name]
            require(member.size == row['bytes'], 'archive member size mismatch')
            h = hashlib.sha256(); size = 0
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(MIB), b''):
                    h.update(block); size += len(block)
            require({'bytes': size, 'sha256': h.hexdigest()} == row, 'archive member hash mismatch')
            seen.add(name)
        # Consume the gzip trailer, including CRC validation.
        while decompressed.read(MIB):
            pass
    require(seen == set(rows), 'all manifest members must survive readback')
    return {'members': len(seen), 'all_members_regular': True, 'all_member_hashes_match_snapshot': True}

def freeze(root, archive, receipt):
    root = Path(root).resolve(); archive = Path(archive).resolve(); receipt = Path(receipt).resolve()
    require(not archive.exists() and not archive.is_symlink() and not receipt.exists()
            and not receipt.is_symlink(), 'fresh archive and receipt destinations required')
    require(archive != receipt and not archive.is_relative_to(root) and not receipt.is_relative_to(root),
            'archive and receipt must be outside immutable snapshot')
    rows = verify_snapshot(root)
    archive.parent.mkdir(parents=True, exist_ok=True); receipt.parent.mkdir(parents=True, exist_ok=True)
    # An incompressible archive cannot require more than payload+member overhead+2MiB.
    allowance = sum(row['bytes'] for row in rows.values()) + 2048*len(rows) + 2*MIB
    require(shutil.disk_usage(archive.parent).free >= RESERVE+allowance,
            '1.5GiB reserve plus worst-case archive allowance required')
    partial = archive.with_name(archive.name+'.partial')
    require(not partial.exists() and not partial.is_symlink(), 'fresh partial archive required')
    with partial.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for name, row in sorted(rows.items()):
                    # Manual TarInfo/addfile is deliberate: filesystem hardlinks
                    # must become independent regular payloads in the public tgz.
                    member = tarfile.TarInfo(PREFIX+name)
                    member.size = row['bytes']; member.mode = 0o644; member.mtime = 0
                    with (root/name).open('rb') as source:
                        tar.addfile(member, source)
        raw.flush(); os.fsync(raw.fileno())
    verified = readback(partial, rows)
    require(verify_snapshot(root) == rows, 'source snapshot changed during archive creation')
    actual = pin(partial)
    os.link(partial, archive); partial.unlink()
    report = {'kind': 'ACTUAL_FINAL_LOCAL_CODE_ARCHIVE_READBACK_V1', 'status': 'PASS',
              'version': '0.5.0', 'archive_filename': archive.name, 'archive': actual,
              'snapshot_manifest': rows['MANIFEST.json'], 'payloads_excluding_manifest': len(rows)-1,
              'archive_prefix': PREFIX, **verified, 'source_snapshot_rehashed_after_archive': True,
              'disk_reserve_bytes': RESERVE, 'external_assets_copied_into_code_archive': False,
              'external_publication_actions': 0, 'remote_actions': 0, 'YT_writes': 0}
    with receipt.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True); stream.write('\n')
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(freeze(args.snapshot, args.archive, args.receipt), indent=2))

if __name__ == '__main__':
    main()
