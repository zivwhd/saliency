
import sys
import os

print("#### IXNN setup ####", file=sys.stderr)
current_dir = os.path.dirname(os.path.abspath(__file__))

targets = [ os.path.join(current_dir, "src")  ]

for path in targets:
    if path not in sys.path:
        sys.path.append(path)
