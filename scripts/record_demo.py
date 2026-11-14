"""Record an actual subprocess stream and render a portable terminal MP4/GIF.

Usage: python scripts/record_demo.py --output docs/media/demo -- command args...
Timing in the cast is original; video playback uses --speed acceleration.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--speed',type=float,default=5)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command:p.error('A command is required')
    dest=Path(a.output); dest.parent.mkdir(parents=True,exist_ok=True)
    header={'version':2,'width':112,'height':30,'timestamp':int(time.time()),'env':{'TERM':'xterm-256color'},
            'command':command,'title':'DPO · actual Apple Silicon training and evaluation'}
    events=[];start=time.monotonic()
    with dest.with_suffix('.cast').open('w') as f:
        f.write(json.dumps(header)+'\n')
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout:
            event=[time.monotonic()-start,'o',line];events.append(event)
            f.write(json.dumps(event)+'\n');f.flush();print(line,end='',flush=True)
        status=proc.wait()
        event=[time.monotonic()-start,'o',f'\nProcess exit code: {status}\n'];events.append(event);f.write(json.dumps(event)+'\n')
    render(dest,events,a.speed)
    if status:sys.exit(status)


def render(dest,events,speed):
    import re,textwrap
    from PIL import Image,ImageDraw,ImageFont
    fontpath='/System/Library/Fonts/Menlo.ttc'
    if not Path(fontpath).exists():fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    font=ImageFont.truetype(fontpath,18);small=ImageFont.truetype(fontpath,15)
    width,height=1280,720; fps=4
    duration=max(8,events[-1][0]/speed+4)
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
                              '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',
                              '-movflags','+faststart',str(dest.with_suffix('.mp4'))],stdin=subprocess.PIPE)
    lines=[];event_index=0;poster=None
    for frame in range(int(duration*fps)):
        elapsed=frame/fps*speed
        while event_index<len(events) and events[event_index][0]<=elapsed:
            for line in events[event_index][2].splitlines():
                line=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',line)
                lines.extend(textwrap.wrap(line,108) or [''])
            event_index+=1
        img=Image.new('RGB',(width,height),'#0b1220');d=ImageDraw.Draw(img)
        d.rounded_rectangle((22,20,1258,698),radius=18,fill='#111c2e',outline='#26364e',width=2)
        for i,color in enumerate(['#ff5f57','#febc2e','#28c840']): d.ellipse((42+i*25,40,55+i*25,53),fill=color)
        d.text((145,34),'DPO / M1 · Recorded terminal output',font=font,fill='#edf5ff')
        d.text((42,78),f'Actual run time {min(elapsed,events[-1][0]):6.1f}s  |  Playback {speed:g}x  |  No simulated metrics',font=small,fill='#8ca8c7')
        for i,line in enumerate(lines[-25:]):
            color='#5eead4' if line.startswith(('===','---','Completed')) else '#d8e4f3'
            d.text((42,116+i*21),line,font=font,fill=color)
        encoder.stdin.write(img.tobytes())
        if event_index==len(events):poster=img
    encoder.stdin.close()
    if encoder.wait()!=0:raise RuntimeError('ffmpeg video encoding failed')
    (poster or img).save(dest.with_suffix('.png'))
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(dest.with_suffix('.mp4')),'-vf',
                    'fps=3,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                    '-loop','0',str(dest.with_suffix('.gif'))],check=True)
    dest.with_suffix('.json').write_text(json.dumps({'source':'actual subprocess stdout/stderr','original_seconds':events[-1][0],
                                                   'playback_speed':speed,'frames_per_second':fps},indent=2)+'\n')

if __name__=='__main__': main()
