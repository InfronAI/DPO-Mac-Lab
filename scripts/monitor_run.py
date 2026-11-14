"""Record process RSS during a run; start immediately after launching it."""
import argparse,csv,time
from pathlib import Path
import psutil
p=argparse.ArgumentParser();p.add_argument('pid',type=int);p.add_argument('output');a=p.parse_args()
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
process=psutil.Process(a.pid);start=time.monotonic()
with path.open('w') as f:
 writer=csv.writer(f);writer.writerow(['elapsed_seconds','rss_bytes','cpu_percent'])
 while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
  try:writer.writerow([time.monotonic()-start,process.memory_info().rss,process.cpu_percent()]);f.flush()
  except psutil.NoSuchProcess:break
  time.sleep(2)
