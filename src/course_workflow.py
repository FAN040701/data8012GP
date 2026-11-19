"""Portable entry points for preparing released AMOS files and reproducing the main comparisons."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = read(path)
        if any(existing.get(k) != v for k,v in value.items()):
            raise ValueError(f'Existing partition differs from release manifest: {path}')
        return
    path.write_text(json.dumps(value, indent=2))


def prepare(source, jobs):
    from prepare_data import prepare_one
    manifest = read(ROOT/'configs/course_cases.json')
    parts, files = manifest['splits'], manifest['files']
    flat = [c for ids in parts.values() for c in ids]
    assert len(flat)==len(set(flat))==len(files)
    index = {c: group for group, ids in parts.items() for c in ids}
    def prepare_case(case):
        entry = files[case]
        member = entry['image']
        for typ, rel in [('image',member), ('label',member.replace('images','labels'))]:
            src = source / rel
            dest = ROOT/'data/raw'/rel
            with src.open('rb') as handle:
                if hashlib.file_digest(handle,'sha256').hexdigest() != entry[typ+'_sha256']:
                    raise ValueError(f'Checksum mismatch: {rel}')
            if src.resolve()!=dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    with dest.open('rb') as f:
                        if hashlib.file_digest(f,'sha256').hexdigest()!=entry[typ+'_sha256']: raise ValueError(dest)
                else:
                    tmp = dest.with_suffix('.partial')
                    shutil.copyfile(src, tmp)
                    tmp.replace(dest)
        return prepare_one((index[case] if index[case] in ('train','val') else 'test', member))
    primary = {k:parts[k] for k in ['train','val','test']}
    meta = ROOT/'data/metadata'
    save(meta/'split.json', {'splits':primary,'members':{k:[files[c]['image'] for c in ids] for k,ids in primary.items()}})
    save(meta/'mechanism_replication/split.json',{'test':parts['replication']})
    save(meta/'continuation/split.json',{'test':parts['prior20']})
    with ThreadPoolExecutor(jobs) as pool:
        for i, m in enumerate(pool.map(prepare_case,flat),1): print(i, m['case'],flush=True)


def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/'src'/script),*map(str,args)],check=True,cwd=ROOT)


def baseline(seed, kind):
    folder = ROOT/'results/runs'/f'{kind}_seed{seed}'
    if not (folder/'complete.json').exists():
        if (folder/'train.jsonl').exists(): raise RuntimeError(f'Incomplete fit at {folder}')
        run('experiment.py','--kind',kind,'--seed',seed)
    if kind != 'repeat': run('evaluate.py','--kind',kind,'--seed',seed)


def controls(seed):
    if seed <= 33:
        baseline(seed,'repeat')
        trainer = 'train_continuation.py'
    else:
        out=ROOT/'results/seed_stability'
        out.mkdir(exist_ok=True,parents=True)
        if not (out/'pipeline_checks.json').exists(): run('check_seed_stability.py')
        trainer='train_seed_stability.py'
    for arm in ['repeat','repeat_aug','context','context_aug']:
        run(trainer,'--seed',seed,'--arm',arm,'--steps',40000)
        from course_controls import evaluate
        evaluate(seed,arm)


def offsets(seed):
    for policy in ['near_far','far_near','mid','far']:
        for suffix in ['', '_aug']:
            run('train_layout.py','--arm',policy+suffix,'--seed',seed)
            run('eval_slice_selection.py','--arm',policy+suffix,'--seed',seed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    data=sub.add_parser('prepare')
    data.add_argument('--source',type=Path,required=True,help='Directory containing the extracted amos22/ folder')
    data.add_argument('--workers',type=int,default=3)
    models=sub.add_parser('baseline')
    models.add_argument('--kind',choices=['2d','2p5d','3d'],required=True)
    models.add_argument('--seed',type=int,choices=[11,22,33],default=11)
    sub.add_parser('analyze')
    for name in ['controls','offsets','sensitivity']:
        sub.add_parser(name).add_argument('--seed',type=int,choices=[11,22,33,44,55,66],required=True)
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.source,args.workers)
    elif args.action=='baseline': baseline(args.seed,args.kind)
    elif args.action=='controls': controls(args.seed)
    elif args.action=='offsets': offsets(args.seed)
    elif args.action=='sensitivity':
        for arm in ['near_far','near_far_aug','far_near','far_near_aug']:
            if args.seed<=33: run('eval_slice_selection.py','--arm',arm,'--seed',args.seed,'--cohort','original')
            run('eval_slice_selection.py','--arm',arm,'--seed',args.seed,'--checkpoint','last')
    else:
        from course_controls import gather
        gather()
        run('analyze_offsets.py')
        run('select_offsets.py')
        run('analyze_offsets.py','--repeat')
        run('analyze_final_study.py')


if __name__=='__main__':
    main()
