#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import subprocess, sys, xml.etree.ElementTree as ET

resources = Path(__file__).resolve().parents[1]
errors=[]
for number in range(29,51):
    matches=list(resources.glob(f'lesson-{number}-*'))
    if len(matches)!=1:
        errors.append(f'lesson {number}: expected one directory, got {len(matches)}'); continue
    lesson=matches[0]; readme=lesson/'README.md'; script=lesson/f'run_lesson{number}.py'
    if not readme.exists(): errors.append(f'{lesson}: missing README.md'); continue
    text=readme.read_text(encoding='utf-8')
    for required in ('代码','常见错误理解','检验问题与参考答案','**参考答案：**','一句话复述'):
        if required not in text: errors.append(f'{readme}: missing {required}')
    if '面试追问' in text: errors.append(f'{readme}: must use neutral self-check wording')
    if '\\[' in text or '\\]' in text: errors.append(f'{readme}: unsupported math delimiter')
    if text.count('```')%2: errors.append(f'{readme}: unbalanced code fences')
    if len(text)<6000: errors.append(f'{readme}: too short ({len(text)} chars)')
    if not script.exists(): errors.append(f'{lesson}: missing run script')
    else:
        c=subprocess.run([sys.executable,'-m','py_compile',str(script)],capture_output=True,text=True)
        if c.returncode: errors.append(f'{script}: compile failed: {c.stderr}')
        r=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,timeout=30)
        if r.returncode: errors.append(f'{script}: run failed: {r.stderr}')
    for svg in lesson.glob('*.svg'):
        try: ET.parse(svg)
        except Exception as exc: errors.append(f'{svg}: invalid SVG: {exc}')
print('Lessons:',22); print('Errors:',len(errors))
for error in errors: print('ERROR:',error)
sys.exit(1 if errors else 0)
