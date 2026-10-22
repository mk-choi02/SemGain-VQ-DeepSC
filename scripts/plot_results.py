"""Plot mean empirical distortion from JSON result (matplotlib optional)."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--out',default='results/semgain.png')
    args=ap.parse_args()
    with open(args.input) as f: data=json.load(f)
    rows=data['rows']
    col='measured_additive_risk' if data['config']['mode']=='toy' else 'measured_delta_ce'
    for policy in sorted(set(r['policy'] for r in rows)):
        xx=sorted(set(float(r['snr_db']) for r in rows))
        yy=[np.mean([r[col] for r in rows if r['policy']==policy and r['snr_db']==x and r[col] is not None]) if any(r['policy']==policy and r['snr_db']==x and r[col] is not None for r in rows) else np.nan for x in xx]
        plt.plot(xx,yy,marker='o',label=policy)
    plt.xlabel('QPSK Es/N0 [dB]');plt.ylabel('Risk CONDITIONAL on header delivery' if col=='measured_additive_risk' else 'CE gap CONDITIONAL on header delivery')
    plt.title('Budget-matched payload; see JSON header_frame_error_rate separately')
    plt.grid(True,alpha=0.2);plt.legend()
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    plt.tight_layout();plt.savefig(args.out,dpi=180)
    print(args.out)

if __name__=='__main__':main()
