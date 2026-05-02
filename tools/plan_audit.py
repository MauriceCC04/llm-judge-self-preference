import argparse, json, glob, os, hashlib, random
from collections import Counter, defaultdict


def load_plans(plans_dir):
    plans=[]
    for path in sorted(glob.glob(os.path.join(plans_dir,'*.json'))):
        if path.endswith('.provenance.json'):
            continue
        with open(path,'r',encoding='utf-8') as f:
            obj=json.load(f)
        plans.append((path,obj))
    return plans


def session_signature(obj):
    days=obj.get('plan',{}).get('days',[])
    return tuple((d.get('session_type'), d.get('duration_minutes'), d.get('is_rest_day'), d.get('is_hard_day')) for d in days)


def text_fingerprint(obj):
    parts=[]
    for d in obj.get('plan',{}).get('days',[]):
        parts.append(str(d.get('title','')))
        parts.append(str(d.get('workout','')))
        parts.append(str(d.get('purpose','')))
    parts.append(str(obj.get('readiness',{}).get('rationale','')))
    return hashlib.sha256('\n'.join(parts).encode('utf-8')).hexdigest()[:16]


def compact_summary(path,obj):
    days=obj.get('plan',{}).get('days',[])
    sess=Counter(d.get('session_type','?') for d in days)
    rest=sum(1 for d in days if d.get('is_rest_day'))
    hard=sum(1 for d in days if d.get('is_hard_day'))
    total=sum(int(d.get('duration_minutes') or 0) for d in days)
    return {
        'file': os.path.basename(path),
        'days': len(days),
        'rest_days': rest,
        'hard_days': hard,
        'total_minutes': total,
        'session_mix': dict(sess),
        'first_titles': [d.get('title','') for d in days[:3]],
        'signature': hash(session_signature(obj)),
        'text_fp': text_fingerprint(obj),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plans-dir', required=True)
    ap.add_argument('--sample-size', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    args=ap.parse_args()
    plans=load_plans(args.plans_dir)
    rng=random.Random(args.seed)
    sample=plans[:] if len(plans)<=args.sample_size else rng.sample(plans,args.sample_size)
    print('SAMPLED_PLANS')
    for path,obj in sample:
        print(json.dumps(compact_summary(path,obj), ensure_ascii=False))
    sig_groups=defaultdict(list)
    txt_groups=defaultdict(list)
    for path,obj in plans:
        sig_groups[session_signature(obj)].append(os.path.basename(path))
        txt_groups[text_fingerprint(obj)].append(os.path.basename(path))
    print('\nEXACT_SESSION_SIGNATURE_DUPLICATES')
    any_dup=False
    for sig, files in sig_groups.items():
        if len(files)>1:
            any_dup=True
            print(len(files), files[:10])
    if not any_dup:
        print('NONE')
    print('\nEXACT_TEXT_FINGERPRINT_DUPLICATES')
    any_dup=False
    for fp, files in txt_groups.items():
        if len(files)>1:
            any_dup=True
            print(len(files), files[:10])
    if not any_dup:
        print('NONE')

if __name__=='__main__':
    main()
