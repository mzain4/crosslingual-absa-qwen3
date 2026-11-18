"""Frozen evaluation of the original six directed domain transfers."""
from pathlib import Path
def command(args):
    from .constants import DOMAINS,LANGUAGES
    from .inference import load_model,run_shard
    import gc,torch,json
    out=Path(args.out)
    # This historical interface accepts the original adapter layout.
    adapters={d:Path(args.adapter_dir)/f'{d}-uabsa' for d in DOMAINS}
    for path in adapters.values():
        if not (path/'adapter_config.json').is_file():raise FileNotFoundError(path)
    for source in DOMAINS:
        for target in DOMAINS:
            if source==target:continue
            for language in LANGUAGES:
                if (out/source/target/(language+'.jsonl')).exists():
                    raise FileExistsError('Use a new output directory; existing predictions are preserved')
    for source,path in adapters.items():
        model,tokenizer=load_model(adapter_path=path)
        for target in DOMAINS:
            if target==source:continue
            for language in LANGUAGES:
                destination=out/source/target/(language+'.jsonl')
                result=run_shard(model,tokenizer,Path(args.checkout),target,language,'uabsa','qlora',destination)
                result['source_domain']=source
                destination.with_suffix('.scores.json').write_text(json.dumps(result,indent=2)+'\n')
                print(source,target,language,result['f1'],flush=True)
        del model,tokenizer
        gc.collect();torch.cuda.empty_cache()
    return 0
def add_parser(sub):
    p=sub.add_parser('historical-transfer',help='original three-domain, six-language frozen UABSA grid')
    p.add_argument('--checkout',default='vendor/M-ABSA')
    p.add_argument('--adapter-dir',required=True)
    p.add_argument('--out',required=True)
    p.set_defaults(func=command)

