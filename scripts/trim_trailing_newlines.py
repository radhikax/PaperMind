from pathlib import Path
files = [Path('src/agents.py'), Path('src/ingest.py'), Path('src/vectorstore.py')]
for p in files:
    s = p.read_text(encoding='utf-8')
    s = s.replace('\r\n', '\n')
    while s.endswith('\n\n'):
        s = s[:-1]
    if not s.endswith('\n'):
        s += '\n'
    p.write_text(s.replace('\n','\r\n'), encoding='utf-8')
    print('fixed', p)
