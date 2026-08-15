from pathlib import Path
files = [Path('src/agents.py'), Path('src/ingest.py'), Path('src/vectorstore.py')]
for p in files:
    b = p.read_bytes()
    while b.find(b'\r\r\n') != -1:
        b = b.replace(b'\r\r\n', b'\r\n')
    p.write_bytes(b)
    print('normalized', p)
