"""Command line interface; all defaults are local and account-free."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='DPO on Apple Silicon: inspect, prepare, run, report, generate, judge')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    p=sub.add_parser('prepare'); p.add_argument('--config',default='experiments/mac.json'); p.add_argument('--output',required=True)
    p=sub.add_parser('run'); p.add_argument('--config',default='experiments/smoke.json'); p.add_argument('--output',required=True)
    p.add_argument('--device',default='auto'); p.add_argument('--data'); p.add_argument('--resume',action='store_true')
    p=sub.add_parser('report'); p.add_argument('directory')
    p=sub.add_parser('generate'); p.add_argument('--model',required=True); p.add_argument('--prompt',required=True)
    p.add_argument('--device',default='auto'); p.add_argument('--max-new-tokens',type=int,default=48)
    p.add_argument('--temperature',type=float,default=0.); p.add_argument('--task',choices=['sentiment','summarization','dialogue'],default='dialogue')
    p=sub.add_parser('judge'); p.add_argument('--answers',required=True); p.add_argument('--key',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('agreement'); p.add_argument('--left',required=True); p.add_argument('--right',required=True); p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='doctor':
        from dpo_lab.runtime import environment, select_device
        print(json.dumps(environment(select_device()),indent=2))
    elif args.command=='prepare':
        from dpo_lab.data import prepare
        from dpo_lab.pipeline import validate_config
        cfg=json.loads(Path(args.config).read_text()); validate_config(cfg)
        print(json.dumps(prepare(cfg,args.output),indent=2))
    elif args.command=='run':
        from dpo_lab.pipeline import run
        try:
            run(args.config,args.output,args.device,args.data,args.resume)
        except Exception as exc:
            from dpo_lab.runtime import write_json
            path=Path(args.output)/'failure.json'
            write_json(path,{'type':type(exc).__name__,'message':str(exc)})
            raise
    elif args.command=='report':
        from dpo_lab.report import report
        report(args.directory)
    elif args.command=='generate':
        from dpo_lab.models import load_model,load_tokenizer
        from dpo_lab.evaluation import generate
        from dpo_lab.runtime import select_device,seed_everything
        device=select_device(args.device); seed_everything(0)
        model=load_model(args.model,device); tok=load_tokenizer(args.model)
        cfg={'max_prompt_length':128,'max_new_tokens':args.max_new_tokens}
        for _,text,_ in generate(model,tok,args.prompt,cfg,device,args.task,args.temperature): print(text)
    elif args.command=='agreement':
        from dpo_lab.judging import agreement
        from dpo_lab.runtime import read_jsonl,write_json
        result=agreement(read_jsonl(args.left),read_jsonl(args.right)); write_json(args.output,result)
        print(json.dumps(result,indent=2))
    elif args.command=='judge':
        from dpo_lab.evaluation import score_judgments
        from dpo_lab.runtime import read_jsonl,write_json
        result=score_judgments(read_jsonl(args.answers),read_jsonl(args.key)); write_json(args.output,result)
        print(json.dumps(result,indent=2))

if __name__=='__main__': main()
