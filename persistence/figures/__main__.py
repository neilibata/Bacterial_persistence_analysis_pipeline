"""
Run: python -m persistence.figures --ref UHGG
     python -m persistence.figures --section s2 --ref UHGG
"""
import sys

# Quick peek at args to route to the right section
_args = sys.argv[1:]
if "--section" in _args:
    idx = _args.index("--section")
    sec = _args[idx + 1] if idx + 1 < len(_args) else ""
    # Remove the --section flag so downstream argparse doesn't choke
    sys.argv = [sys.argv[0]] + _args[:idx] + _args[idx+2:]
    if sec == "s2":
        from .section2 import main
    else:
        from .section1 import main
else:
    from .section1 import main

main()
