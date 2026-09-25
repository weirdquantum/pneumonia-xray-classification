"""Read-only image audit and deterministic group-stratified split. No training."""
import argparse
import csv
import hashlib
import json
import platform
import random
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import PIL
from PIL import Image

LABELS = {'NORMAL': 0, 'PNEUMONIA': 1}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
POLICY = 'cnn-filename-rgb-content-v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def proxy_key(name, relative):
    """These namespaces are filename conventions, not verified patient IDs."""
    stem = Path(name).stem.lower()
    match = re.fullmatch(r'person(\d+)_(?:bacteria|virus)_\d+(?:_\d+)*', stem)
    if match:
        return 'pneumonia-person:' + str(int(match[1])), True
    match = re.fullmatch(r'(normal2-im|im)-(\d+)-\d+(?:-\d+)*', stem)
    if match:
        return match[1] + ':' + str(int(match[2])), True
    return 'unrecognized:' + relative, False


def scan(root):
    root = Path(root).resolve()
    for split in ('train', 'test'):
        for label in LABELS:
            if not (root / split / label).is_dir():
                raise ValueError(f'Missing directory: {split}/{label}; use a root containing train/ and test/.')
    rows, ignored = [], []
    for split in ('train', 'test'):
        unexpected = [p.name for p in (root / split).iterdir()
                      if p.is_dir() and p.name not in LABELS and not p.name.startswith('.')]
        if unexpected:
            raise ValueError(f'Unexpected class directories in {split}: {unexpected}')
        for label in LABELS:
            candidates = sorted((root / split / label).rglob('*'))
            image_count = 0
            for p in candidates:
                rel = p.relative_to(root).as_posix()
                if p.is_symlink():
                    raise ValueError(f'Symlinks are not accepted: {rel}')
                if not p.is_file():
                    continue
                if any(part.startswith('.') or part == '__MACOSX' for part in p.relative_to(root).parts) or p.suffix.lower() not in EXTENSIONS:
                    ignored.append(rel)
                    continue
                image_count += 1
                key, recognized = proxy_key(p.name, rel)
                row = dict(path=rel, original_split=split, class_name=label, label=LABELS[label],
                           bytes=p.stat().st_size, raw_sha256='', pixel_sha256='', width=0, height=0,
                           mode='', proxy_group=key, recognized_proxy=recognized, group_id='',
                           status='eligible', reason='', split='')
                try:
                    row['raw_sha256'] = digest(p.read_bytes())
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', Image.DecompressionBombWarning)
                        with Image.open(p) as image:
                            image.verify()
                        with Image.open(p) as image:
                            image.load()
                            row.update(width=image.width, height=image.height, mode=image.mode)
                            rgb = image.convert('RGB')
                            row['pixel_sha256'] = digest(f'RGB:{rgb.width}:{rgb.height}:'.encode() + rgb.tobytes())
                except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                    row.update(status='quarantine', reason='decode_error:' + type(exc).__name__)
                rows.append(row)
            if not image_count:
                raise ValueError(f'No image files found in {split}/{label}')
    return rows, ignored


def assign_groups(rows):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    seen = {}
    for i, row in enumerate(rows):
        for field in ('proxy_group', 'raw_sha256', 'pixel_sha256'):
            if not row[field]:
                continue
            key = field, row[field]
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    for members in components.values():
        gid = digest('\n'.join(sorted(r['path'] for r in members)).encode())
        reasons = []
        if len({r['original_split'] for r in members}) > 1:
            reasons.append('cross_original_split_group')
        if len({r['label'] for r in members}) > 1:
            reasons.append('conflicting_labels_group')
        # A corrupt image may be a member of a known proxy group. Isolate the whole group.
        if any(r['status'] == 'quarantine' for r in members):
            reasons.append('group_contains_decode_error')
        hashes = set()
        for row in sorted(members, key=lambda r: r['path']):
            row['group_id'] = gid
            if reasons:
                row.update(status='quarantine', reason=';'.join(filter(None, [row['reason'], *reasons])))
            elif row['pixel_sha256'] in hashes:
                row.update(status='quarantine', reason='duplicate_pixels_within_original_split')
            else:
                hashes.add(row['pixel_sha256'])
    return rows


def split_groups(rows, seed=42, val_fraction=0.2):
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between 0 and 1')
    for row in rows:
        if row['status'] == 'eligible' and row['original_split'] == 'test':
            row['split'] = 'development_test'
    for label in LABELS.values():
        groups = defaultdict(list)
        for row in rows:
            if row['status'] == 'eligible' and row['original_split'] == 'train' and row['label'] == label:
                groups[row['group_id']].append(row)
        if len(groups) < 2:
            raise ValueError(f'Class {label} has fewer than two eligible training groups')
        ordered = sorted(groups)
        random.Random(seed + label).shuffle(ordered)
        total = sum(map(len, groups.values()))
        target = total * val_fraction
        # Reachable subset sums produce the nearest feasible per-class image ratio.
        # Immutable predecessor tuples prevent reusing a group during reconstruction.
        reachable = {0: None}
        cap = min(total - 1, target + max(map(len, groups.values())))
        for gid in ordered:
            size = len(groups[gid])
            for count, chain in list(reachable.items()):
                new = count + size
                if new <= cap and new not in reachable:
                    reachable[new] = (gid, chain)
        candidates = [n for n in reachable if 0 < n < total]
        if not candidates:
            raise ValueError(f'Cannot construct nonempty train/validation for class {label}')
        best = min(candidates, key=lambda n: (abs(n - target), n))
        selected = set()
        chain = reachable[best]
        while chain is not None:
            gid, chain = chain
            selected.add(gid)
        for gid, members in groups.items():
            for row in members:
                row['split'] = 'val' if gid in selected else 'train'
    verify(rows)
    return rows


def verify(rows):
    for split in ('train', 'val', 'development_test'):
        if {r['label'] for r in rows if r['split'] == split} != {0, 1}:
            raise ValueError(f'{split} must contain both classes')
    for field in ('group_id', 'raw_sha256', 'pixel_sha256'):
        locations = defaultdict(set)
        for r in rows:
            if r['split']:
                if r['status'] != 'eligible' or not r[field]:
                    raise ValueError('Invalid included row')
                locations[r[field]].add(r['split'])
        if any(len(parts) > 1 for parts in locations.values()):
            raise ValueError(f'Leakage found for {field}')


def audit(root, out, seed=42, val_fraction=0.2):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty; use a new directory to preserve the frozen split')
    rows, ignored = scan(root)
    rows = split_groups(assign_groups(rows), seed, val_fraction)
    identity = [{k: r[k] for k in ('path', 'raw_sha256', 'pixel_sha256')} for r in rows]
    partitions = [{k: r[k] for k in ('path', 'group_id', 'status', 'reason', 'split')} for r in rows]
    summary = dict(schema_version=1, policy=POLICY, seed=seed, val_fraction_requested=val_fraction,
                   patient_level_verified=False, near_duplicate_detection=False,
                   evaluation_name='original test set development evaluation',
                   dataset_sha256=digest(json_bytes(identity)), split_sha256=digest(json_bytes(partitions)),
                   script_sha256=digest(Path(__file__).read_bytes()), python=platform.python_version(), pillow=PIL.__version__,
                   observed_images=len(rows), original_counts=dict(Counter(r['original_split']+'/'+r['class_name'] for r in rows)),
                   final_counts=dict(Counter(r['split']+'/'+r['class_name'] for r in rows if r['split'])),
                   quarantine_counts=dict(Counter(r['reason'] for r in rows if r['status']=='quarantine')),
                   unrecognized_filename_count=sum(not r['recognized_proxy'] for r in rows),
                   ignored_files=ignored, excluded_root_entries=sorted(p.name for p in Path(root).iterdir() if p.name not in ('train','test')),
                   notebook_reference_counts={'train':5216,'test':624})
    summary['matches_notebook_total_counts'] = all(sum(r['original_split']==s for r in rows)==n for s,n in summary['notebook_reference_counts'].items())
    out.mkdir(parents=True, exist_ok=True)
    for filename, selected in [('manifest.csv', rows), ('quarantine.csv', [r for r in rows if r['status']=='quarantine'])]:
        with (out/filename).open('w', newline='', encoding='utf-8') as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(selected)
    summary['manifest_sha256']=digest((out/'manifest.csv').read_bytes())
    (out/'summary.json').write_bytes(json_bytes(summary))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--val-fraction',default=0.2,type=float)
    args=parser.parse_args()
    try:
        result=audit(args.data_root,args.output,args.seed,args.val_fraction)
    except ValueError as exc:
        parser.exit(2, str(exc)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
