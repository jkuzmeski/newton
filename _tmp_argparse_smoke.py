"""Parse the exact argv of one experiment run through the real parser, without training."""
import sys
from projects.impedance_instron.bilateral.grid import experiments as exp
from importlib import import_module

run = exp.get(sys.argv[1])[int(sys.argv[2] if len(sys.argv) > 2 else 0)]
argv = ["--dataset", "this/dataset/does/not/exist", "--output", "outputs/_argparse_smoke", *run.args]
print("argv:", " ".join(argv))
try:
    import_module(run.entry).main(argv)
except SystemExit as exc:
    print("SystemExit:", exc)
except Exception as exc:
    print("passed argparse, failed later:", type(exc).__name__, str(exc)[:200])
