import ast
import os
import re
import random
import inspect

# Config 
MAX_PATHS = 800
LOOP_UNROLL_DEPTH = 2
Z3_TIMEOUT_MS = 2000
DEFAULT_INT_RANGE = (-3, 3)
DEFAULT_FLOAT_RANGE = (-3.0, 3.0)
DEFAULT_SAMPLE_PER_PARAM = 4

OUT_DIR = "ticfg_out"


# utility 
def safe_unparse(node):
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__

def ensure_outdir(OUT_DIR="ticfg_out"):
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR

# docstring parsing 
def parse_docstring_domains(func):
    doc = inspect.getdoc(func)
    if not doc:
        return {}
    lines = [ln.rstrip() for ln in doc.splitlines() if ln.strip()]
    domains = {}
    start = None
    for i,l in enumerate(lines):
        if l.strip().lower().startswith("parameters"):
            start = i+1; break
    if start is None: start = 0
    i = start
    while i < len(lines):
        ln = lines[i].lstrip()
        m = re.match(r"(\w+)\s*\((\w+)\)\s*:\s*(.*)", ln)
        if m:
            name, typ, rest = m.groups()
            typ = typ.lower()
            if typ == "int":
                rng = re.search(r"range\[\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\]", rest)
                if rng:
                    a,b = int(float(rng.group(1))), int(float(rng.group(2)))
                    vals = list(range(a, b+1))
                else:
                    vals = list(range(DEFAULT_INT_RANGE[0], DEFAULT_INT_RANGE[1]+1))
                domains[name] = {"type":"int","values":vals}
            elif typ == "float":
                rng = re.search(r"range\[\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\]", rest)
                if rng:
                    a,b = float(rng.group(1)), float(rng.group(2))
                    vals = [round(a + random.random()*(b-a),4) for _ in range(DEFAULT_SAMPLE_PER_PARAM)]
                else:
                    a,b = DEFAULT_FLOAT_RANGE
                    vals = [round(a + random.random()*(b-a),4) for _ in range(DEFAULT_SAMPLE_PER_PARAM)]
                domains[name] = {"type":"float","values":vals}
            elif typ == "bool":
                setm = re.search(r"\{(.+?)\}", rest)
                if setm:
                    items = [it.strip() for it in setm.group(1).split(",")]
                    vals = [True if it.lower()=="true" else False if it.lower()=="false" else it for it in items]
                else:
                    vals = [True, False]
                domains[name] = {"type":"bool","values":vals}
            elif typ == "str":
                setm = re.search(r"\{(.+?)\}", rest)
                if setm:
                    items = [it.strip().strip('"').strip("'") for it in setm.group(1).split(",")]
                    domains[name] = {"type":"str","values":items}
                else:
                    domains[name] = {"type":"str","values":["", "a", "test"]}
            else:
                domains[name] = {}
        i += 1
    return domains

def domain_map_to_samples(domain_map, param_names):
    samples = {}
    for p in param_names:
        spec = domain_map.get(p, {})
        if not spec:
            samples[p] = list(range(DEFAULT_INT_RANGE[0], DEFAULT_INT_RANGE[1]+1))
        else:
            t = spec.get("type")
            if t == "int":
                samples[p] = spec.get("values", list(range(DEFAULT_INT_RANGE[0], DEFAULT_INT_RANGE[1]+1)))
            elif t == "float":
                samples[p] = spec.get("values", [round(random.uniform(DEFAULT_FLOAT_RANGE[0], DEFAULT_FLOAT_RANGE[1]),4) for _ in range(DEFAULT_SAMPLE_PER_PARAM)])
            elif t == "bool":
                samples[p] = spec.get("values", [True, False])
            elif t == "str":
                samples[p] = spec.get("values", ["", "a", "test"])
            else:
                samples[p] = list(range(DEFAULT_INT_RANGE[0], DEFAULT_INT_RANGE[1]+1))
    return samples