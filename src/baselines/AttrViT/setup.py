
import sys
import os

print("#### AttrVit setup ####", file=sys.stderr)
current_dir = os.path.dirname(os.path.abspath(__file__))

targets = [ current_dir  ]

for path in targets:
    if path not in sys.path:
        sys.path.append(path)
