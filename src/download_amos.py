"""Fetch an auditable, seeded AMOS CT subset directly from the official ZIP.

HTTP byte ranges avoid downloading unlabelled test cases or MRI volumes.
Each extracted member is verified against the ZIP CRC by Python's zipfile.
"""
import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import random
import re
import struct
import time
import zipfile
import zlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://zenodo.org/records/7262581/files/amos22.zip'
DOWNLOAD_URL = URL


class HTTPFile(io.RawIOBase):
    def __init__(self, url=URL):
        self.url = url
        self.session = requests.Session()
        r = self.session.get(url, headers={'Range': 'bytes=-65536'}, timeout=120)
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError('Server must support byte ranges')
        self.size = int(r.headers['Content-Range'].split('/')[-1])
        self.position = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        if size == 0:
            return b''
        end = min(self.position + size, self.size) - 1
        for attempt in range(6):
            try:
                r = self.session.get(self.url, headers={'Range': f'bytes={self.position}-{end}'}, timeout=180)
                r.raise_for_status()
                if r.status_code != 206 or len(r.content) != end-self.position+1:
                    raise RuntimeError(f'Invalid range response {r.status_code}: {len(r.content)}')
                self.position = end + 1
                return r.content
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)


def fetch_member(entry):
    name, expected_size, expected_crc, offset, compressed_size, method = entry
    out = ROOT / 'data/raw' / name
    if out.is_file() and out.stat().st_size == expected_size:
        payload = out.read_bytes()
        if (zlib.crc32(payload) & 0xffffffff) == expected_crc:
            return name, expected_size, hashlib.sha256(payload).hexdigest(), 'cached'
    for attempt in range(4):
        try:
            # One HTTP request per member instead of repeatedly reading the ZIP index.
            end = offset + compressed_size + 1023
            r=requests.get(DOWNLOAD_URL,headers={'Range':f'bytes={offset}-{end}'},timeout=(30,180))
            r.raise_for_status()
            if r.status_code != 206 or not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                raise RuntimeError('Incorrect byte range response')
            blob=r.content
            if blob[:4] != b'PK\x03\x04':
                raise RuntimeError('Invalid local ZIP header')
            nlen,elen=struct.unpack_from('<HH',blob,26)
            start=30+nlen+elen
            packed=blob[start:start+compressed_size]
            if len(packed) != compressed_size:
                raise RuntimeError('Incomplete member payload')
            if method == 8:
                payload=zlib.decompress(packed,-15)
            elif method == 0:
                payload=packed
            else:
                raise RuntimeError(f'Unsupported ZIP method {method}')
            if len(payload) != expected_size or (zlib.crc32(payload)&0xffffffff) != expected_crc:
                raise RuntimeError('ZIP member size/CRC mismatch')
            out.parent.mkdir(parents=True, exist_ok=True)
            temp = out.with_suffix(out.suffix+'.partial')
            temp.write_bytes(payload)
            temp.replace(out)
            return name, len(payload), hashlib.sha256(payload).hexdigest(), 'downloaded'
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt+1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--train', type=int, default=80)
    ap.add_argument('--val', type=int, default=20)
    ap.add_argument('--test', type=int, default=40)
    ap.add_argument('--mirror', action='store_true')
    args = ap.parse_args()
    global DOWNLOAD_URL
    if args.mirror:
        DOWNLOAD_URL='https://huggingface.co/datasets/sathiiii/amos22/resolve/main/amos22.zip'
    meta = ROOT / 'data/metadata'
    meta.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(HTTPFile()) as z:
        infos = z.infolist()
        entries = [{'name':i.filename,'size':i.file_size,'crc32':i.CRC,'compressed_size':i.compress_size,'header_offset':i.header_offset,'method':i.compress_type} for i in infos]
        (meta/'archive_index.json').write_text(json.dumps(entries,indent=2))
        names = {i.filename for i in infos}
        image_members = [i.filename for i in infos if re.search(r'/images(?:Tr|Va)/amos_\d+\.nii\.gz$', i.filename)]
        def eligible(folder):
            return sorted(n for n in image_members if f'/{folder}/' in n and int(re.search(r'amos_(\d+)',n).group(1)) < 500 and n.replace('images','labels') in names)
        training = eligible('imagesTr')
        validation = eligible('imagesVa')
        print('Eligible CT pairs:', len(training), 'train;', len(validation), 'released validation', flush=True)
        if len(training) < args.train+args.val or len(validation) < args.test:
            raise RuntimeError('Insufficient expected labelled CT pairs; inspect archive index')
        rng = random.Random(20261114)
        rng.shuffle(training)
        rng.shuffle(validation)
        selected = {'train': training[:args.train], 'val': training[args.train:args.train+args.val], 'test':validation[:args.test]}
        split = {key:[Path(n).name.removesuffix('.nii.gz') for n in ns] for key,ns in selected.items()}
        manifest = {'seed':20261114,'source':URL,'download_url':DOWNLOAD_URL,'archive_bytes':24234336519,'official_archive_md5':'67717b2a483ac0744c89c3016b7aaef7','whole_archive_md5_verified':False,'verification':'Per-member official ZIP CRC32 and local SHA256; archive downloaded selectively','eligible_official_train_ct':len(training),'eligible_official_val_ct':len(validation),'splits':split,'members':selected}
        (meta/'split.json').write_text(json.dumps(manifest,indent=2))
        # Round-robin: enough fit/dev cases arrive early for pipeline smoke checks.
        order = []
        for idx in range(max(map(len,selected.values()))):
            for key in ['train','val','test']:
                if idx < len(selected[key]):
                    n=selected[key][idx]
                    order.extend([n.replace('images','labels'),n])
        lookup={i.filename:i for i in infos}
        tasks=[(n,lookup[n].file_size,lookup[n].CRC,lookup[n].header_offset,lookup[n].compress_size,lookup[n].compress_type) for n in order]
        for name in names:
            if name.endswith('.json') and lookup[name].file_size < 1000000:
                try:
                    (meta/Path(name).name).write_bytes(z.read(name))
                except Exception as e:
                    print('Metadata warning', name, repr(e), flush=True)
    print('Selected patients:', {k:len(v) for k,v in split.items()}, 'files:',len(tasks),'GiB:',round(sum(t[1] for t in tasks)/2**30,2),flush=True)
    started=time.time()
    records=[]
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for fut in cf.as_completed([pool.submit(fetch_member,t) for t in tasks]):
            name,size,sha,status=fut.result()
            records.append({'member':name,'bytes':size,'sha256':sha,'status':status})
            (meta/'download_checksums.json').write_text(json.dumps(records,indent=2))
            print(f'{len(records)}/{len(tasks)} {name} {size/1e6:.1f}MB {status} elapsed={time.time()-started:.1f}s',flush=True)
    for filename in ['labeled_data_meta_0000_0599.csv']:
        r=requests.get('https://zenodo.org/records/7262581/files/'+filename,timeout=90)
        r.raise_for_status()
        (meta/filename).write_bytes(r.content)
    (meta/'download_complete.json').write_text(json.dumps({'files':len(records),'elapsed_seconds':time.time()-started},indent=2))


if __name__ == '__main__':
    main()
