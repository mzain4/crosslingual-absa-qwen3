"""Repeated training and frozen transfer using the recorded study protocols.

Model libraries are imported only in an explicitly requested GPU worker. Each
model load runs in its own process; plans and numerical checks are CPU-only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager

from . import constants as C


@dataclass(frozen=True)
class Job:
    model: str
    domain: str
    task: str
    seed: int
    empty_weight: float = 1.0

    @property
    def name(self):
        arm='original' if self.empty_weight==1 else 'weighted025'
        return f'{self.model}-{self.domain}-{self.task}-seed{self.seed}-{arm}'


def protocol(job):
    ids={'qwen':(C.MODEL_ID,C.MODEL_REVISION),'mt5':(C.MT5_MODEL_ID,C.MT5_REVISION),
         'gemma':(C.GEMMA_MODEL_ID,C.GEMMA_REVISION)}
    model,revision=ids[job.model]
    common={**asdict(job),'model_id':model,'model_revision':revision,'dataset_revision':C.MABSA_REVISION}
    if job.model=='mt5':
        return {**common,'epochs':30,'max_steps_cap':2500,'micro_batch':4,'effective_batch':16,
            'learning_rate':3e-4,'warmup_ratio':.1,'weight_decay':0,'adam_epsilon':1e-6,
            'source_length':128,'target_length':128,'precision':'fp32',
            'selection':'English development raw F1 every 100 updates during final 500, plus final'}
    return {**common,'epochs':3,'micro_batch':1,'effective_batch':8,'learning_rate':2e-4,
        'answer_serialization':'recorded_tuple_text_to_json',
        'scheduler':'linear','warmup_steps':0,'weight_decay':0,'max_length':1536,
        'rank':8,'alpha':16,'dropout':.05,'target_modules':list(C.LORA_TARGET_MODULES),
        'optimizer':'paged_adamw_8bit','selection':'final three-epoch checkpoint',
        'max_new_tokens':C.MAX_NEW_TOKENS[job.task]}


def jobs_for(args):
    if args.checkpoint_every < 1:raise ValueError('Checkpoint interval must be positive')
    domains=args.domains or (['laptop'] if args.model=='gemma' else
        ['food','phone'] if args.empty_weight!=1 or args.cross_domain else list(C.DOMAINS))
    tasks=args.tasks or (['uabsa'] if args.model=='gemma' or args.empty_weight!=1 or args.cross_domain else list(C.TASKS))
    seeds=args.seeds or list(C.REPEATED_SEEDS)
    if set(domains)-set(C.DOMAINS) or set(tasks)-set(C.TASKS) or set(seeds)-set(C.REPEATED_SEEDS):
        raise ValueError('Use the recorded domains, tasks and seeds 13, 42, 87')
    if args.empty_weight not in (1.0,.25):raise ValueError('Recorded empty weights are 1 and 0.25')
    if args.empty_weight!=1 and (args.model!='qwen' or set(domains)-{'food','phone'} or tasks!=['uabsa']):
        raise ValueError('Recorded weighting covers Qwen Food/Phone UABSA')
    if args.model=='gemma' and (domains!=['laptop'] or tasks!=['uabsa']):
        raise ValueError('The recorded Gemma comparison is Laptop UABSA')
    if args.cross_domain and (args.model!='qwen' or tasks!=['uabsa'] or set(domains)-{'food','phone'}):
        raise ValueError('Repeated transfer uses Qwen Food/Phone UABSA')
    if args.cross_domain and args.stage=='train':
        raise ValueError('Transfer reuses existing adapters; omit --cross-domain for training')
    return [Job(args.model,d,t,s,args.empty_weight) for d in domains for t in tasks for s in seeds]


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def freeze_manifest(path,value):
    path=Path(path)
    if path.exists() and read_json(path)!=value:
        raise ValueError(f'Existing run identity differs: {path}. Use a new output directory.')
    write_json(path,value)


def verify_checkout(checkout):
    result=subprocess.run(['git','rev-parse','HEAD'],cwd=checkout,capture_output=True,text=True,check=True)
    if result.stdout.strip()!=C.MABSA_REVISION:
        raise ValueError('Dataset revision differs; use clabsa prepare first')
    changed=subprocess.run(['git','diff','--quiet','HEAD','--','data'],cwd=checkout,check=False)
    if changed.returncode:raise ValueError('Tracked dataset files were modified')


def environment_versions():
    from importlib.metadata import version,PackageNotFoundError
    result={}
    for name in ('torch','transformers','peft','bitsandbytes','accelerate'):
        try:result[name]=version(name)
        except PackageNotFoundError:result[name]=None
    return result


@contextmanager
def job_lock(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'worker.lock'
    try:stream=path.open('x',encoding='utf-8')
    except FileExistsError as error:
        raise RuntimeError(f'Job is locked: {path}. Remove only after confirming its previous worker has stopped.') from error
    try:
        with stream:stream.write(str(os.getpid()))
        yield
    finally:path.unlink(missing_ok=True)


def prediction_prefix(path):
    """Return valid records and their byte length; only a final torn line may recover."""
    path=Path(path)
    if not path.exists():return [],0
    lines=path.read_bytes().splitlines(keepends=True);rows=[];length=0
    for index,line in enumerate(lines):
        try:row=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if index!=len(lines)-1 or line.endswith(b'\n'):raise ValueError(f'Corrupt saved prediction at line {index+1}')
            break
        if not isinstance(row,dict):raise ValueError('Saved prediction must be an object')
        rows.append(row);length+=len(line)
    return rows,length


def commit_prefix_repair(path,length):
    """After identity checks, preserve any incomplete tail and append on a new line."""
    path=Path(path)
    if not path.exists():return
    payload=path.read_bytes()
    if length<len(payload):
        import uuid
        path.with_name(path.name+'.incomplete-'+uuid.uuid4().hex).write_bytes(payload[length:])
        with path.open('r+b') as stream:stream.truncate(length);stream.flush();os.fsync(stream.fileno())
    if length and not payload[:length].endswith(b'\n'):
        with path.open('ab') as stream:stream.write(b'\n');stream.flush();os.fsync(stream.fileno())


def scale_review_loss(loss,weight):
    if weight.numel()!=1:raise ValueError('Review weighting requires microbatch one')
    return loss*weight.to(loss.device).reshape(())


def weighted_trainer_class(base):
    class ReviewTrainer(base):
        total_visits=0
        weighted_visits=0
        def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
            weight=inputs.pop('review_weight')
            self.total_visits+=1
            self.weighted_visits+=float(weight.item())==.25
            result=super().compute_loss(model,inputs,return_outputs=return_outputs,**kwargs)
            if return_outputs:
                loss,output=result
                return scale_review_loss(loss,weight),output
            return scale_review_loss(result,weight)
    return ReviewTrainer


def _prepare_causal(job,adapter=None):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
    from peft import LoraConfig,PeftModel,get_peft_model,prepare_model_for_kbit_training
    spec=protocol(job);set_seed(job.seed)
    tokenizer=AutoTokenizer.from_pretrained(spec['model_id'],revision=spec['model_revision'])
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if job.model=='gemma':
        if not torch.cuda.is_bf16_supported():raise RuntimeError('Gemma protocol requires BF16 support')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
    model=AutoModelForCausalLM.from_pretrained(spec['model_id'],revision=spec['model_revision'],
        device_map={'':0},dtype=dtype,quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype))
    model.config.use_cache=False
    if adapter is None:model=prepare_model_for_kbit_training(model)
    if job.model=='gemma':
        # Preserve frozen matrix dtype at training and reload, as in the v5 run.
        for parameter in model.parameters():
            if type(parameter).__name__=='Params4bit':continue
            if parameter.is_floating_point():
                parameter.data=parameter.data.to(torch.bfloat16 if parameter.ndim>=2 else torch.float32)
        if adapter is None:set_seed(job.seed)
    if adapter is not None:
        model=PeftModel.from_pretrained(model,str(adapter));model.config.use_cache=True;model.eval()
    else:
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',
            task_type='CAUSAL_LM',target_modules=list(C.LORA_TARGET_MODULES)))
        expected=17_440_768 if job.model=='gemma' else C.ADAPTER_TRAINABLE_PARAMS
        count=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count!=expected:raise ValueError(f'Adapter size {count} differs from recorded {expected}')
    return model,tokenizer


def _training(job,args):
    import torch
    from transformers import Trainer,TrainingArguments,TrainerCallback,Seq2SeqTrainer,Seq2SeqTrainingArguments
    from .data import read_split
    from .train_qlora import encode_example,AnswerCollator
    from .parsing import to_official_string
    from .scoring import score_strings
    folder=Path(args.out)/job.name;folder.mkdir(parents=True,exist_ok=True)
    examples=read_split(Path(args.checkout),job.domain,'en','train')
    spec=protocol(job)
    source_hash=digest(Path(args.checkout)/'data'/job.domain/'en'/'train.txt')
    freeze_manifest(folder/'configuration.json',{**spec,'training_data_sha256':source_hash,
        'environment':environment_versions()})
    if (folder/'training_complete.json').exists():
        committed=read_json(folder/'training_complete.json')
        if not all(digest(folder/name)==value for name,value in committed['files'].items()):
            raise ValueError('Completed model integrity check failed')
        print('SKIP completed',job.name,flush=True);return
    steps=min(2500,math.ceil(len(examples)/16)*30) if job.model=='mt5' else math.ceil(len(examples)/8)*3
    class Recovery(TrainerCallback):
        def on_step_end(self,arguments,state,control,**kwargs):
            if state.global_step==steps:control.should_save=True
            if job.model=='mt5':
                evaluate=state.global_step>=max(1,steps-500) and (state.global_step%100==0 or state.global_step==steps)
                control.should_evaluate=evaluate
                control.should_save=evaluate or state.global_step%args.checkpoint_every==0
            return control
        def on_evaluate(self,arguments,state,control,metrics=None,**kwargs):
            if job.model!='mt5':return control
            score=float(metrics['eval_raw_f1'])
            best=read_json(folder/'best.json') if (folder/'best.json').exists() else None
            if best is None or score>best['raw_f1']:
                destination=folder/f'best-step-{state.global_step}'
                trainer.save_model(str(destination))
                tokenizer.save_pretrained(destination)
                write_json(folder/'best.json',{'step':state.global_step,'raw_f1':score,
                    'directory':destination.name,'files':{p.name:digest(p) for p in destination.iterdir() if p.is_file()}})
            return control
        def on_log(self,arguments,state,control,logs=None,**kwargs):
            if logs:print(job.name,state.global_step,'/',steps,logs,flush=True)
        def on_save(self,arguments,state,control,**kwargs):
            checkpoint=folder/'checkpoints'/f'checkpoint-{state.global_step}'
            if checkpoint.exists():
                paths=[p for p in checkpoint.iterdir() if p.is_file() and p.name!='complete.json']
                write_json(checkpoint/'complete.json',{'step':state.global_step,
                    'files':{p.name:digest(p) for p in paths},
                    'total_visits':getattr(trainer,'total_visits',0),'weighted_visits':getattr(trainer,'weighted_visits',0)})
    base_args=dict(output_dir=str(folder/'checkpoints'),seed=job.seed,data_seed=job.seed,
        report_to='none',remove_unused_columns=False,logging_steps=25,dataloader_num_workers=0,
        disable_tqdm=True,save_total_limit=2,ignore_data_skip=False)
    if job.model=='mt5':
        import numpy as np
        from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,default_data_collator,set_seed
        set_seed(job.seed)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        tokenizer=AutoTokenizer.from_pretrained(C.MT5_MODEL_ID,revision=C.MT5_REVISION)
        model=AutoModelForSeq2SeqLM.from_pretrained(C.MT5_MODEL_ID,revision=C.MT5_REVISION,dropout_rate=.1)
        model.config.use_cache=False
        dev=read_split(Path(args.checkout),job.domain,'en','dev')
        def encode(rows):
            result=[]
            for row in rows:
                source=tokenizer(row.sentence,padding='max_length',truncation=True,max_length=128)
                target=tokenizer(to_official_string(row.targets(job.task),job.task),padding='max_length',truncation=True,max_length=128)
                result.append({**source,'labels':[v if v!=tokenizer.pad_token_id else -100 for v in target['input_ids']],
                    'decoder_attention_mask':target['attention_mask']})
            return result
        gold=[to_official_string(row.targets(job.task),job.task) for row in dev]
        def metrics(value):
            ids=value.predictions[0] if isinstance(value.predictions,tuple) else value.predictions
            ids=np.where(ids<0,tokenizer.pad_token_id,ids)
            predicted=tokenizer.batch_decode(ids,skip_special_tokens=True)
            return {'raw_f1':score_strings(predicted,gold,job.task)['f1']}
        arguments=Seq2SeqTrainingArguments(**base_args,num_train_epochs=30,max_steps=steps,
            learning_rate=3e-4,per_device_train_batch_size=4,gradient_accumulation_steps=4,
            per_device_eval_batch_size=8,warmup_ratio=.1,weight_decay=0,adam_epsilon=1e-6,
            eval_strategy='no',save_strategy='steps',save_steps=args.checkpoint_every,
            load_best_model_at_end=False,gradient_checkpointing=True,optim='adamw_torch',
            lr_scheduler_type='linear',predict_with_generate=True,
            generation_max_length=128,generation_num_beams=1,fp16=False,bf16=False)
        trainer=Seq2SeqTrainer(model=model,args=arguments,train_dataset=encode(examples),eval_dataset=encode(dev),
            data_collator=default_data_collator,processing_class=tokenizer,compute_metrics=metrics,callbacks=[Recovery()])
    else:
        model,tokenizer=_prepare_causal(job)
        data=[]
        for row in examples:
            item=encode_example(tokenizer,job.task,job.domain,row.sentence,row.targets(job.task))
            item['review_weight']=job.empty_weight if not row.targets(job.task) else 1.0
            data.append(item)
        collator=AnswerCollator(tokenizer)
        def weighted_collator(features):
            batch=collator(features)
            batch['review_weight']=torch.tensor([row['review_weight'] for row in features])
            return batch
        arguments=TrainingArguments(**base_args,num_train_epochs=3,learning_rate=2e-4,
            per_device_train_batch_size=1,gradient_accumulation_steps=8,lr_scheduler_type='linear',
            warmup_steps=0,weight_decay=0,optim='paged_adamw_8bit',gradient_checkpointing=True,
            bf16=torch.cuda.is_bf16_supported(),fp16=not torch.cuda.is_bf16_supported(),
            save_strategy='steps',save_steps=args.checkpoint_every)
        trainer=weighted_trainer_class(Trainer)(model=model,args=arguments,train_dataset=data,
            data_collator=weighted_collator,callbacks=[Recovery()])
    write_json(folder/'training_arguments.json',arguments.to_dict())
    resume=None
    for candidate in sorted((folder/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]),reverse=True):
        if not (candidate/'complete.json').exists():continue
        commit=read_json(candidate/'complete.json')
        if all((candidate/name).is_file() and digest(candidate/name)==value for name,value in commit['files'].items()):
            resume=str(candidate)
            trainer.total_visits=commit.get('total_visits',0);trainer.weighted_visits=commit.get('weighted_visits',0)
            break
    if resume is None and any((folder/'checkpoints').glob('checkpoint-*')):
        raise RuntimeError('Checkpoint directories exist but none verifies; refusing to restart training silently')
    if resume and read_json(Path(resume)/'trainer_state.json')['global_step']>=steps:
        trainer._load_from_checkpoint(resume)
        trainer.state.global_step=steps
    else:trainer.train(resume_from_checkpoint=resume)
    if job.model=='mt5':
        if not (folder/'best.json').exists():raise ValueError('English development selection is missing')
        best=read_json(folder/'best.json');selected=folder/best['directory']
        if not all((selected/name).is_file() and digest(selected/name)==value for name,value in best['files'].items()):
            raise ValueError('Selected development model is incomplete; preserve checkpoints and investigate')
        model=AutoModelForSeq2SeqLM.from_pretrained(selected)
    final=folder/'final';model.save_pretrained(final);tokenizer.save_pretrained(final)
    files={str(p.relative_to(folder)):digest(p) for p in final.rglob('*') if p.is_file()}
    write_json(folder/'training_complete.json',{'step':trainer.state.global_step,'files':files,
        'total_visits':getattr(trainer,'total_visits',None),'weighted_visits':getattr(trainer,'weighted_visits',None),
        'best_checkpoint':read_json(folder/'best.json') if job.model=='mt5' else None})
    print('COMPLETE',job.name,flush=True)


def _evaluation(job,args):
    import torch
    from .data import read_split
    from .prompts import build_messages
    from .inference import generate
    from .parsing import to_official_string,process_prediction
    from .categories import categories_for
    from .scoring import score_strings,to_percentage
    base=Path(args.adapter_root or args.out)/job.name
    commit=read_json(base/'training_complete.json')
    if not all(digest(base/name)==value for name,value in commit['files'].items()):raise ValueError('Model files changed')
    final=base/'final'
    if job.model=='mt5':
        from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
        tokenizer=AutoTokenizer.from_pretrained(final);model=AutoModelForSeq2SeqLM.from_pretrained(final).to('cuda').eval()
        model.config.use_cache=True
    else:model,tokenizer=_prepare_causal(job,final)
    targets=[d for d in ('food','phone') if d!=job.domain] if args.cross_domain else [job.domain]
    languages=args.languages or (['en','sw'] if args.cross_domain else list(C.LANGUAGES))
    if set(languages)-set(C.LANGUAGES):raise ValueError('Unknown evaluation language')
    for target in targets:
        for language in languages:
            folder=Path(args.out)/job.name/'predictions'/target
            folder.mkdir(parents=True,exist_ok=True)
            identity={**protocol(job),'target_domain':target,'language':language,'model_files':commit['files']}
            freeze_manifest(folder/(language+'-identity.json'),identity)
            path=folder/(language+'.jsonl')
            examples=read_split(Path(args.checkout),target,language,'test')
            rows,prefix_bytes=prediction_prefix(path)
            if len(rows)>len(examples):raise ValueError('Saved prediction prefix exceeds target split')
            for index,row in enumerate(rows):
                if row['source_index']!=index or row['sentence']!=examples[index].sentence or row['gold_text']!=to_official_string(examples[index].targets(job.task),job.task):
                    raise ValueError('Saved prediction prefix differs from pinned data')
            commit_prefix_repair(path,prefix_bytes)
            with path.open('a',encoding='utf-8') as stream:
                for index in range(len(rows),len(examples)):
                    example=examples[index]
                    if job.model=='mt5':
                        batch=tokenizer(example.sentence,return_tensors='pt',truncation=True,max_length=128).to('cuda')
                        with torch.inference_mode():output=model.generate(**batch,do_sample=False,num_beams=1,max_length=128)
                        prediction=tokenizer.decode(output[0],skip_special_tokens=True);raw=prediction;status='text-to-text'
                    else:
                        raw=generate(model,tokenizer,build_messages(job.task,target,example.sentence),C.MAX_NEW_TOKENS[job.task])
                        prediction,status=process_prediction(raw,job.task,categories_for(target) if job.task=='tasd' else ())
                    row={'source_index':index,'sentence':example.sentence,
                        'gold_text':to_official_string(example.targets(job.task),job.task),'raw_output':raw,
                        'prediction_text':prediction,'status':status}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno());rows.append(row)
                    if (index+1)%25==0:print(job.name,target,language,index+1,'/',len(examples),flush=True)
            metric=to_percentage(score_strings([r['prediction_text'] for r in rows],[r['gold_text'] for r in rows],job.task))
            write_json(folder/(language+'-scores.json'),{**identity,'n':len(rows),**metric,'prediction_sha256':digest(path)})
            print('SCORED',job.name,target,language,metric,flush=True)


def command(args):
    jobs=jobs_for(args)
    if args.stage=='plan':
        print(json.dumps({'jobs':[protocol(job) for job in jobs],'cross_domain':args.cross_domain},indent=2));return 0
    if args.worker:
        verify_checkout(args.checkout)
        job=Job(**read_json(args.worker))
        with job_lock(Path(args.out)/job.name):
            import torch
            if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required for this stage')
            if args.stage=='train':_training(job,args)
            else:_evaluation(job,args)
        return 0
    for job in jobs:
        request=Path(args.out)/'queue'/(job.name+'-'+args.stage+'.json');write_json(request,asdict(job))
        argv=[sys.executable,'-m','clabsa','experiment','--model',args.model,'--stage',args.stage,
            '--checkout',args.checkout,'--out',args.out,'--empty-weight',str(args.empty_weight),
            '--checkpoint-every',str(args.checkpoint_every),'--worker',str(request)]
        for name in ('domains','tasks','seeds','languages'):
            values=getattr(args,name)
            if values:argv+=['--'+name,*map(str,values)]
        if args.cross_domain:argv.append('--cross-domain')
        if args.adapter_root:argv+=['--adapter-root',args.adapter_root]
        env={**os.environ,'PYTHONHASHSEED':str(job.seed),'CUBLAS_WORKSPACE_CONFIG':':4096:8',
            'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
        result=subprocess.run(argv,env=env)
        if result.returncode:return result.returncode
    return 0


def add_parser(subparsers):
    parser=subparsers.add_parser('experiment',help='plan, train or evaluate the repeated model comparisons')
    parser.add_argument('--model',choices=['qwen','mt5','gemma'],default='qwen')
    parser.add_argument('--stage',choices=['plan','train','evaluate'],default='plan')
    parser.add_argument('--checkout',default='vendor/M-ABSA')
    parser.add_argument('--out',default='runs/repeated')
    parser.add_argument('--adapter-root')
    parser.add_argument('--domains',nargs='+')
    parser.add_argument('--tasks',nargs='+')
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--languages',nargs='+')
    parser.add_argument('--empty-weight',type=float,choices=[1,.25],default=1)
    parser.add_argument('--cross-domain',action='store_true')
    parser.add_argument('--checkpoint-every',type=int,default=100)
    parser.add_argument('--worker',help=__import__('argparse').SUPPRESS)
    parser.set_defaults(func=command)
