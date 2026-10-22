"""Convert common pre-split text JSON to EHeum/DeepSC's train/test/vocab format.

Never reorder an existing vocabulary! This is a schema converter, NOT a new
preprocessing or tokenizer implementation. Retains validation in a sidecar.
"""
import argparse
import json
from pathlib import Path

SPECIALS=['<PAD>','<START>','<END>','<UNK>']

def _sentences(x,name):
    if isinstance(x,dict):
        for field in ('sentences','texts','text','data'):
            if field in x: return _sentences(x[field],name)
    if isinstance(x,list):
        parsed=[]
        for y in x:
            if isinstance(y,str):parsed.append(y)
            elif isinstance(y,dict):
                if 'sentence' in y:parsed.append(y['sentence'])
                elif 'text' in y:parsed.append(y['text'])
                else:raise ValueError(f'{name}: unknown example dict {sorted(y.keys())}')
            else:raise ValueError(f'{name}: expected strings or dictionaries')
        if not all(isinstance(s,str) and s.strip() for s in parsed):
            raise ValueError(f'{name}: empty or non-string sentence')
        return parsed
    raise ValueError(f'{name} must be a list or {{sentences:[...]}}')

def _vocab(v):
    if isinstance(v,dict):
        for key in ('itos','tokens','id_to_token','id2token'):
            if key in v:return _vocab(v[key])
        # String token -> integer ID; rebuild only by explicitly stored IDs.
        if all(isinstance(n,int) for n in v.values()):
            pairs=sorted(v.items(),key=lambda kv:kv[1]);v=[w for w,_ in pairs]
        elif all(str(k).isdigit() for k in v.keys()):
            v=[w for _,w in sorted(v.items(),key=lambda kv:int(kv[0]))]
    if not isinstance(v,list) or not all(isinstance(w,str) for w in v):
        raise ValueError('Cannot read ordered vocabulary; provide --vocab-source DeepSC corpus')
    if len(set(v))!=len(v) or v[:4]!=SPECIALS:
        raise ValueError('Vocabulary order or four special tokens do not match DeepSC')
    return v

def convert(input_path,output_path,vocab_source=None):
    d=json.loads(Path(input_path).read_text(encoding='utf-8'))
    src=d.get('splits',d)
    train=_sentences(src['train'],'train')
    test=_sentences(src['test'],'test')
    val=_sentences(src['val' if 'val' in src else 'validation'],'val') if ('val' in src or 'validation' in src) else []
    if len(set(train))!=len(train) or len(set(test))!=len(test) or (val and len(set(val))!=len(val)):
        raise ValueError('Duplicate examples within at least one split')
    if set(train)&set(test) or set(train)&set(val) or set(test)&set(val):
        raise ValueError('Exact normalized sentence leakage across splits')
    v=d.get('vocab',d.get('vocabulary'))
    if vocab_source is not None:
        ref=json.loads(Path(vocab_source).read_text(encoding='utf-8'))
        v=ref.get('vocab',ref.get('vocabulary'))
    if v is None:raise ValueError('Input lacks explicit vocabulary; supply --vocab-source from pretrained corpus')
    v=_vocab(v)
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'train':train,'test':test,'vocab':v},ensure_ascii=False),encoding='utf-8')
    if val:
        out.with_suffix('.validation.json').write_text(json.dumps({'validation':val},ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'out':str(out),'train':len(train),'val_reserved':len(val),'test':len(test),
                      'vocabulary_size':len(v)},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--vocab-source',default=None)
    a=ap.parse_args();convert(a.input,a.out,a.vocab_source)
