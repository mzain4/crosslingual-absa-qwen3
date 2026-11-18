"""Verify the committed numerical records without models or a GPU."""
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics
import hashlib


def verify(root):
    root=Path(root)
    if (root/'SHA256.json').exists():
        for name,expected in json.loads((root/'SHA256.json').read_text()).items():
            with (root/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=expected:raise ValueError(f'File hash mismatch: {name}')
    scores=json.loads((root/'scores.json').read_text(encoding='utf-8'))
    conditions=json.loads((root/'conditions.json').read_text(encoding='utf-8'))
    totals={};row_totals=Counter()
    with gzip.open(root/'review_counts.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            row_totals[row['condition']]+=1
            totals.setdefault(row['condition'],Counter()).update({key:row[key] for key in
                ('tp','predicted','gold','multiset_tp','omission','empty_false_positive','empty_gold')})
    if len({r['condition'] for r in scores})!=len(scores) or set(totals)!={r['condition'] for r in scores}:
        raise ValueError('Condition records are duplicated or missing')
    if {r['id'] for r in conditions}!=set(totals):raise ValueError('Condition identities differ')
    for condition in conditions:
        if row_totals[condition['id']]!=condition['rows']:raise ValueError('Condition row count differs')
    for row in scores:
        counts=totals[row['condition']]
        if any(counts[key]!=row[key] for key in counts):
            raise ValueError(f'Count mismatch in {row["condition"]}')
        denominator=counts['predicted']+counts['gold']
        f1=200*counts['tp']/denominator if denominator else 0
        if abs(f1-row['raw_f1'])>1e-10:
            raise ValueError(f'F1 mismatch in {row["condition"]}')
        multiset=200*counts['multiset_tp']/denominator if denominator else 0
        if abs(multiset-row['multiset_f1'])>1e-10:raise ValueError('Multiset F1 mismatch')
    def mean(**filters):
        return statistics.mean(row['raw_f1'] for row in scores if all(row[k]==v for k,v in filters.items()))
    result={'verified_conditions':len(scores),'verified_review_records':sum(row_totals.values()),'means':{}}
    for model in ('qwen','mt5'):
        for task in ('tasd','uabsa'):
            result['means'][f'{model}_{task}']=mean(scope='within',model=model,task=task,split='test',arm='control')
    result['means']['gemma_laptop_uabsa']=mean(model='gemma')
    for domain in ('food','phone'):
        for arm in ('control','weighted'):
            result['means'][f'{domain}_{arm}']=mean(scope='within',model='qwen',task='uabsa',target_domain=domain,split='test',arm=arm)
    for arm in ('control','weighted'):
        result['means'][f'cross_domain_{arm}']=mean(scope='cross',arm=arm)
    return result


def add_parser(subparsers):
    parser=subparsers.add_parser('verify-recorded',help='verify committed repeated-study counts and scores on CPU')
    parser.add_argument('--results',default='outputs/repeated-study')
    parser.set_defaults(func=lambda args:print(json.dumps(verify(args.results),indent=2)) or 0)
