
import sys
import os

print("#### GIG setup ####", file=sys.stderr)
current_dir = os.path.dirname(os.path.abspath(__file__))

targets = [ os.path.join(current_dir)  ]

for path in targets:
    if path not in sys.path:
        print(f"adding {path}")
        sys.path.append(path)
