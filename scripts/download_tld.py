"""
Download the external TLD (Joker5800/Tobacco_Leaf_Diseases) test images via the
hf-mirror.com endpoint (huggingface.co is network-blocked in this environment).

Reads data/tld_test_manifest.json (1392 relpaths: 232 distinct base + 1160 = 5x
augmented). Downloads each with hf_hub_download (cached), copies into a
self-contained local tree data/tld_images/test/<class>/<file>.jpg, and writes
data/tld_local_manifest.json = [{relpath, local, class, is_base, size}].

Resumable: skips files already present locally.
"""
import os, json, shutil
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import concurrent.futures as cf
from huggingface_hub import hf_hub_download

ROOT   = ROOT
REPO   = 'Joker5800/Tobacco_Leaf_Diseases'
MAN    = f'{ROOT}/data/tld_test_manifest.json'
LOCAL  = f'{ROOT}/data/tld_images'
OUT    = f'{ROOT}/data/tld_local_manifest.json'
WORKERS = 8

def fetch(relpath, _retries=3):
    local = os.path.join(LOCAL, relpath)          # data/tld_images/test/<class>/<file>
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return {'relpath': relpath, 'local': local, 'cached': True}
    last = None
    for _ in range(_retries):
        try:
            src = hf_hub_download(repo_id=REPO, filename=relpath, repo_type='dataset')
            os.makedirs(os.path.dirname(local), exist_ok=True)
            shutil.copyfile(src, local)
            return {'relpath': relpath, 'local': local, 'cached': False,
                    'size': os.path.getsize(local)}
        except Exception as e:
            last = e
    raise RuntimeError(f'failed {relpath}: {last}')

def main():
    rel = json.load(open(MAN))
    print(f'{len(rel)} files to ensure; workers={WORKERS}')
    results, ok, fail = [], 0, 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, r): r for r in rel}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = futs[fut]
            try:
                results.append(fut.result()); ok += 1
            except Exception as e:
                fail += 1; print('  FAIL', r, str(e)[:120])
            if i % 100 == 0 or i == len(rel):
                print(f'  {i}/{len(rel)}  ok={ok} fail={fail}')
    results.sort(key=lambda x: x['relpath'])
    total_bytes = sum(x.get('size', 0) for x in results)
    base = sum(1 for x in results if '_aug' not in os.path.basename(x['relpath']))
    print(f'\nDONE ok={ok} fail={fail}  base={base}  ~{total_bytes/1e6:.1f} MB')
    json.dump(results, open(OUT, 'w'), indent=2)
    print('saved', OUT)

if __name__ == '__main__':
    main()
