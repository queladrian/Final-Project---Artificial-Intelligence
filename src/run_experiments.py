import argparse, subprocess, sys, json, csv
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--out',default='results'); p.add_argument('--stage',choices=['legacy','stability','nine'],default='stability'); p.add_argument('--iters',type=int,default=15000); p.add_argument('--threads',type=int,default=2); a=p.parse_args()
trials=['50al1'] if a.stage!='nine' else [f'{t}al{i}' for t in (50,75,100) for i in (1,2,3)]
for trial in trials:
 for k in ([100,200] if a.stage=='stability' else [100]):
  for seed in (range(5) if a.stage=='stability' else [0]):
   domain='legacy' if a.stage=='legacy' else 'full'; folder=Path(a.out)/f'{trial}_k{k}_seed{seed}_{domain}_it{a.iters}_s5'
   if (folder/'summary.json').exists(): continue
   subprocess.run([sys.executable,str(Path(__file__).with_name('pinn_inverse.py')),str(Path(a.data)/f'{trial}.csv'),'--k0',str(k),'--seed',str(seed),'--iters',str(a.iters),'--domain',domain,'--out',a.out,'--threads',str(a.threads)],check=True)
rows=[json.loads(f.read_text()) for f in sorted(Path(a.out).glob('*/summary.json'))]
if rows:
 with (Path(a.out)/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
