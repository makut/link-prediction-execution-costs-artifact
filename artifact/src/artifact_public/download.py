"""Bounded public-source downloads. Prints a plan unless --execute is supplied."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import urllib.request

SOURCES = {
    'citation2': [dict(name='citation-v2.zip', bytes=2293454318,
                      url='https://snap.stanford.edu/ogb/data/linkproppred/citation-v2.zip')],
    'friendster': [dict(name=name, bytes=size, md5=checksum,
                       url=f'https://zenodo.org/api/records/5146230/files/{name}/content')
                   for name, size, checksum in (
                       ('com-friendster-dense.ez', 6572926076, '81896fdbaccd0f41608ffc9065433c7f'),
                       ('com-friendster-dense.vz', 99315813, 'a07ac60fe42f7ab9e50c7224edc6e3c1'),
                       ('com-friendster-dense.properties', 1178, '72388acdf5c735aff65d23359370dcf9'))],
}


def digests(path):
    hashes = {name: hashlib.new(name) for name in ('sha256', 'md5')}
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            for digest in hashes.values():
                digest.update(block)
    return {name: digest.hexdigest() for name, digest in hashes.items()}


def download(item, target):
    """Never replace an existing file; bounded streaming and checksum verification."""
    if target.exists():
        receipt = target.with_name(target.name + '.download.json')
        if not receipt.is_file():
            raise FileExistsError(f'existing file has no verified download receipt: {target}')
        prior = json.loads(receipt.read_text())
        actual = digests(target)
        if (prior.get('url') != item['url'] or target.stat().st_size != item['bytes']
                or prior.get('sha256') != actual['sha256']
                or (item.get('md5') and actual['md5'] != item['md5'])):
            raise ValueError(f'existing download identity mismatch: {target}')
        return prior
    partial = target.with_name(target.name + '.partial')
    total = 0
    hashes = {name: hashlib.new(name) for name in ('sha256', 'md5')}
    request = urllib.request.Request(item['url'], headers={'User-Agent': 'LPP-public-artifact/1'})
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get('Content-Length')
        if length is not None and int(length) != item['bytes']:
            raise ValueError('public source size changed; inspect metadata before proceeding')
        with partial.open('xb') as output:
            while True:
                block = response.read(min(4 << 20, item['bytes'] - total + 1))
                if not block:
                    break
                total += len(block)
                if total > item['bytes']:
                    raise ValueError('download exceeded pinned size')
                output.write(block)
                for digest in hashes.values():
                    digest.update(block)
    if total != item['bytes']:
        raise ValueError(f'incomplete download: {total}/{item["bytes"]}; partial file retained')
    result = {**item, **{name: digest.hexdigest() for name, digest in hashes.items()}}
    result['source_kind'] = 'fresh_official_public_download'
    if item.get('md5') and result['md5'] != item['md5']:
        raise ValueError('Zenodo provider MD5 mismatch; partial file retained')
    partial.rename(target)
    result['verification'] = ('provider size+MD5; locally computed SHA256' if item.get('md5')
                              else 'provider size only; locally computed SHA256, no provider SHA256 available')
    target.with_name(target.name + '.download.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=SOURCES)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-download-gib', type=float, default=8)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    items = SOURCES[args.dataset]
    total = sum(item['bytes'] for item in items)
    if (not math.isfinite(args.max_download_gib) or args.max_download_gib <= 0
            or total > args.max_download_gib * (1 << 30)):
        parser.error('dataset exceeds explicit download budget')
    print(json.dumps({'dataset': args.dataset, 'total_bytes': total, 'files': items,
                      'execute': args.execute}, indent=2), flush=True)
    if not args.execute:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    missing = sum(item['bytes'] for item in items if not (args.out / item['name']).exists())
    if shutil.disk_usage(args.out).free < missing + (1 << 30):
        raise RuntimeError('insufficient free disk: payloads plus 1GiB reserve required')
    for item in items:
        print(json.dumps(download(item, args.out / item['name'])), flush=True)


if __name__ == '__main__':
    main()
