
import importlib
import json
import sys


# -----------------------------------------
# KNOBS
# Change any setting from the command line, without editing code:
#   python pipeline/run_onion.py --set=onion.BORDER_THRESHOLD=0.6 --set=onion.POINTING_THRESHOLD=0.45
#   python pipeline/run_onion.py '--set=onion.POINTING_DIRECTION={"lexical":0.3,"entity":0.2,"llm":0.4,"emphasis":0.1}'
# The name is <file without .py>.<CONSTANT>. Values are JSON (numbers, lists, objects) or plain text.
# -----------------------------------------

def apply(argv=None):

    argv = sys.argv if argv is None else argv

    for token in [t for t in argv if t.startswith("--set=")]:

        name, _, raw = token[len("--set="):].partition("=")
        module_name, _, const = name.partition(".")

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            raise SystemExit(f"knob {name}: there is no file {module_name}.py in pipeline/")

        if not hasattr(module, const) or not const.isupper():
            known = sorted(n for n in dir(module) if n.isupper())
            raise SystemExit(f"knob {name}: {module_name} has no such setting. Settings: {', '.join(known)}")

        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw

        old = getattr(module, const)

        # A whole-number value for a decimal setting is fine; anything else must keep its kind
        if isinstance(old, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)

        if type(value) is not type(old):
            raise SystemExit(f"knob {name}: expected {type(old).__name__}, got {type(value).__name__}")

        setattr(module, const, value)

        print(f"knob {module_name}.{const} = {value}  (was {old})")

        argv.remove(token)
