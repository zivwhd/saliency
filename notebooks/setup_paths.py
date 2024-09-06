# setup_paths.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
targets = [
    #os.path.join(current_dir, 'RISE'),
    os.path.join(current_dir, '..', 'src'),
    "c:\\projects\\CPE\\pytorch-grad-cam"
]

for path in targets:
    if path not in sys.path:
        sys.path.append(path)

os.chdir(os.path.join(current_dir, '..'))

