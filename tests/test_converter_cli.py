import json
import subprocess
import sys
from pathlib import Path
import pytest
from scripts.convert_europarl import convert

VOCAB=['<PAD>','<START>','<END>','<UNK>','hello','world']

def test_split_conversion_preserves_validation_and_vocab(tmp_path):
    src=tmp_path/'source.json';out=tmp_path/'converted.json'
    src.write_text(json.dumps({'train':['hello world'], 'validation':['world hello'],
                               'test':['hello hello'],'vocab':VOCAB}))
    convert(src,out)
    result=json.loads(out.read_text())
    assert result['vocab']==VOCAB
    assert result['train']==['hello world']
    assert result['test']==['hello hello']
    assert json.loads(out.with_suffix('.validation.json').read_text())['validation']==['world hello']


def test_convert_rejects_split_leakage(tmp_path):
    inp=tmp_path/'dup.json'
    inp.write_text(json.dumps({'train':['same'], 'test':['same'],'vocab':VOCAB}))
    with pytest.raises(ValueError,match='leakage'):
        convert(inp,tmp_path/'out.json')


def test_actual_cli_emits_budget_safe_json(tmp_path):
    root=Path(__file__).resolve().parents[1]
    dest=tmp_path/'results.json'
    subprocess.run([sys.executable,'scripts/run_semgain.py','--mode','toy',
                    '--snrs','6','--length','8','--trials','3',
                    '--group-size','2','--out',str(dest)],
                   cwd=root,check=True,capture_output=True,text=True)
    assert dest.exists()
    rows=json.loads(dest.read_text())['rows']
    assert len(rows)==4
    assert all(r['total_symbols']<=r['budget'] for r in rows)
    assert all('header_frame_error_rate' in r for r in rows)
