#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys, xml.etree.ElementTree as ET

resources = Path(__file__).resolve().parents[1]
errors = []
for number in range(18, 29):
    matches = list(resources.glob(f'lesson-{number}-*'))
    if len(matches) != 1:
        errors.append(f'lesson {number}: expected one directory, got {len(matches)}')
        continue
    lesson = matches[0]
    readme = lesson / 'README.md'
    script = lesson / f'run_lesson{number}.py'
    if not readme.exists(): errors.append(f'{lesson}: missing README.md')
    if not script.exists(): errors.append(f'{lesson}: missing run script')
    if readme.exists():
        text = readme.read_text(encoding='utf-8')
        if '\\[' in text or '\\]' in text:
            errors.append(f'{readme}: contains unsupported \\[ / \\] math delimiter')
        if text.count('```math') == 0 and any(k in text for k in ('公式', '计算', '概率', '长度')):
            errors.append(f'{readme}: expected at least one fenced math block')
        if text.count('```') % 2:
            errors.append(f'{readme}: unbalanced fenced code block')
    for svg in lesson.glob('*.svg'):
        try: ET.parse(svg)
        except Exception as exc: errors.append(f'{svg}: invalid SVG: {exc}')
    if script.exists():
        compiled = subprocess.run([sys.executable, '-m', 'py_compile', str(script)], capture_output=True, text=True)
        if compiled.returncode: errors.append(f'{script}: compile failed: {compiled.stderr}')
        ran = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=20)
        if ran.returncode: errors.append(f'{script}: run failed: {ran.stderr}')
print('Lessons:', 11)
print('Errors:', len(errors))
for error in errors: print('ERROR:', error)
sys.exit(1 if errors else 0)
