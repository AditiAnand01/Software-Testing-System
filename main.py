"""
Interprocedural Graph-Based Tester with Z3 symbolic pruning.

Usage:
  main.py --file path/to/file.py
  main.py --dir path/to/project_dir
"""
import ast
import os
import sys
import json
import csv
import re
import runpy
import inspect
import types
import random
import itertools
import traceback
from collections import defaultdict
import argparse
import networkx as nx
import matplotlib.pyplot as plt

from utils import safe_unparse, ensure_outdir, parse_docstring_domains, domain_map_to_samples
from z3_solver import try_solve_with_z3
from cfg import CFGBuilder, build_icfg, enumerate_icfg_paths, visualize_icfg, compute_metrics
from visualize import visualize_function_cfgs, build_function_call_graph, visualize_function_call_graph

# Try z3
try:
    import z3
    HAS_Z3 = True
except Exception:
    HAS_Z3 = False

# Config 
MAX_PATHS = 800
LOOP_UNROLL_DEPTH = 2
Z3_TIMEOUT_MS = 2000
DEFAULT_INT_RANGE = (-3, 3)
DEFAULT_FLOAT_RANGE = (-3.0, 3.0)
DEFAULT_SAMPLE_PER_PARAM = 4

OUT_DIR = "ticfg_out"  # output CSVs/JSON saved here


# conditions 
def extract_conditions(icfg, path):
    conds = []
    for n in path:
        lbl = icfg.nodes[n].get("label","")
        if lbl.startswith("If:") or lbl.startswith("LoopCond:") or lbl.startswith("ForLoop"):
            cond = lbl.split(":",1)[1].strip() if ":" in lbl else lbl
            conds.append(cond)
    return conds

def heuristic_solve(conditions, param_names):
    sol = {}
    for cond in conditions:
        m = re.match(r"(\w+)\s*([<>=!]+)\s*([-\d\.]+)", cond)
        if not m:
            continue
        var, op, val = m.groups()
        valn = int(float(val))
        if op in (">", ">="):
            sol[var] = valn + 1
        elif op in ("<", "<="):
            sol[var] = valn - 1
        elif op == "==":
            sol[var] = valn
        elif op == "!=":
            sol[var] = valn + 1
    for p in param_names:
        if p not in sol:
            sol[p] = 0
    return sol


# trace-run helpers 
def _get_func_filename(func):
    # get the underlying code object's filename for functions / methods
    try:
        # bound method: has __func__
        if hasattr(func, "__func__"):
            return func.__func__.__code__.co_filename
        return func.__code__.co_filename
    except Exception:
        return None

def run_function_with_trace(func, kwargs):
    """
    Run a function while tracing executed line numbers, but only record lines
    from the function's defining file (to avoid recording library code).
    Returns (result, exception_or_None, set(of executed line numbers)).
    """
    executed_lines = set()
    target_file = _get_func_filename(func)

    def tracer(frame, event, arg):
        if event == "line":
            try:
                fname = frame.f_code.co_filename
                if target_file is None or os.path.abspath(fname) == os.path.abspath(target_file):
                    executed_lines.add(frame.f_lineno)
            except Exception:
                pass
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        res = func(**kwargs)
        exc = None
    except Exception as e:
        res = None
        exc = e
    finally:
        sys.settrace(old)
    return res, exc, executed_lines

def run_module_as_script_with_trace(path, argv):
    """
    Run the module as a script (runpy.run_path) and trace lines only in the module file.
    Returns (stdout_text, stderr_text, exception_or_None, executed_lines).
    """
    import io
    old_stdout, old_stderr, old_argv = sys.stdout, sys.stderr, sys.argv[:]
    sys.argv = [path] + argv
    sys.stdout = io.StringIO(); sys.stderr = io.StringIO()
    executed_lines = set()
    module_path = os.path.abspath(path)

    def tracer(frame, event, arg):
        if event == "line":
            try:
                fname = os.path.abspath(frame.f_code.co_filename)
                if fname == module_path:
                    executed_lines.add(frame.f_lineno)
            except Exception:
                pass
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    exc = None
    try:
        runpy.run_path(path, run_name="__main__")
    except Exception as e:
        exc = e
    finally:
        sys.settrace(old)
        out_text, err_text = sys.stdout.getvalue(), sys.stderr.getvalue()
        sys.stdout, sys.stderr, sys.argv = old_stdout, old_stderr, old_argv
    return out_text, err_text, exc, executed_lines


# map lines -> nodes/edges 
def build_lineno_map(func_cfgs):
    mapping = defaultdict(list)
    for owner, g in func_cfgs.items():
        for n,d in g.nodes(data=True):
            ln = d.get("lineno")
            if ln is not None:
                # store tuple (owner, node) if you need later; keep node id for now
                mapping[ln].append(n)
    return mapping

def infer_covered_nodes_edges(executed_lines, func_cfgs, icfg):
    mapping = build_lineno_map(func_cfgs)
    covered_nodes = set()
    for ln in executed_lines:
        covered_nodes.update(mapping.get(ln, []))
    covered_edges = set()
    for u,v in icfg.edges():
        if u in covered_nodes and v in covered_nodes:
            covered_edges.add((u,v))
    return covered_nodes, covered_edges


def _safe_repr(x):
    try:
        return json.dumps(x)
    except Exception:
        return repr(x)


# main pipeline 
def analyze_and_test_file(path):
    outdir = ensure_outdir(OUT_DIR)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    icfg, func_cfgs, func_entries, func_ast_map = build_icfg(src, loop_unroll=LOOP_UNROLL_DEPTH)
    print(f"[+] ICFG built: nodes={icfg.number_of_nodes()}, edges={icfg.number_of_edges()}")

    paths = enumerate_icfg_paths(icfg, max_paths=MAX_PATHS)
    print(f"[+] Enumerated {len(paths)} candidate interprocedural paths (capped).")

    # import module dynamically into a dict, with proper __file__ and __name__
    module = {"__file__": os.path.abspath(path), "__name__": "__analyzed_module__"}
    try:
        exec(compile(src, path, "exec"), module)
    except Exception as e:
        print("Warning importing target module via exec failed:", e)
        try:
            module = runpy.run_path(path)
        except Exception as e2:
            print("Warning importing target module via runpy failed too:", e2)

    # collect functions and classes with docstring domains
    func_param_names = {}
    parsed_domains = {}

    # collect normal functions
    for name, obj in module.items():
        if isinstance(obj, types.FunctionType):
            try:
                func_param_names[name] = list(inspect.signature(obj).parameters.keys())
            except Exception:
                func_param_names[name] = []
            parsed_domains[name] = parse_docstring_domains(obj)

    # collect class methods
    for name, obj in module.items():
        if isinstance(obj, type):
            for mname, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                full_name = f"{name}.{mname}"
                try:
                    func_param_names[full_name] = list(inspect.signature(method).parameters.keys())
                except Exception:
                    func_param_names[full_name] = []
                parsed_domains[full_name] = parse_docstring_domains(method)

    # prune infeasible paths using Z3 and generate feasible inputs
    path_tests = []
    for i, p in enumerate(paths, start=1):
        conds = extract_conditions(icfg, p)
        params = set()
        # collect params by scanning node owners like "Owner::label" or "Class.method::label"
        for n in p:
            if "::" in n:
                owner = n.split("::", 1)[0]
                # owner may be "funcname" or "Class.method" or "Class" etc
                if owner in func_param_names:
                    params.update(func_param_names[owner])
                else:
                    # try to resolve "Class.method" -> "Class.method"
                    if "." in owner:
                        nm = owner
                        if nm in func_param_names:
                            params.update(func_param_names[nm])
                        else:
                            # sometimes owner might be "Class" and method nodes later specify; skip
                            pass
        params = list(params)
        solved_by = None
        sol = None
        infeasible = False
        if params and HAS_Z3:
            try:
                sol = try_solve_with_z3(conds, params)
                if sol is None:
                    infeasible = True
                    solved_by = "z3_unsat_or_unsupported"
            except Exception as e:
                print(f"[!] Z3 solving failed for path {i}: {e}")
                sol = None
                infeasible = False

        if not sol:
            # fallback: sample from docstrings or heuristics
            domain_map = {}
            for pnm in params:
                dm = {}
                # try to find domain in any function that uses this param
                for fname, plist in func_param_names.items():
                    if pnm in plist:
                        dm = parsed_domains.get(fname, {}).get(pnm, {})
                        break
                domain_map[pnm] = dm
            samples = domain_map_to_samples(domain_map, params)
            # pick first sample for each param if exists else 0
            sol = {p: (samples.get(p, [0])[0] if p in samples else 0) for p in params}
            solved_by = "sampled" if any(domain_map.values()) else "heuristic"
            infeasible = False

        path_tests.append({
            "path_id": i,
            "path": p,
            "conditions": conds,
            "params": params,
            "inputs": sol or {},
            "solved_by": solved_by,
            "feasible": not infeasible
        })

    # write path_tests.csv
    path_csv = os.path.join(outdir, "path_tests.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path_id", "feasible", "solved_by", "params", "inputs", "conditions"])
        for pt in path_tests:
            w.writerow([
                pt["path_id"],
                pt["feasible"],
                pt["solved_by"] or "",
                json.dumps(pt["params"]),
                _safe_repr(pt["inputs"]),
                json.dumps(pt["conditions"])
            ])
    print(f"[+] Wrote path tests to {path_csv}")

    # Execute tests (unit, integration, system)
    all_tests, unit_tests, integration_tests, system_tests = [], [], [], []
    executed_lines_total = set()
    test_id = 0

    # helper to detect static/class/instance methods robustly
    def _method_kind(cls, mname):
        """
        Returns one of: 'staticmethod', 'classmethod', 'function' (instance method), 'unknown'
        """
        try:
            desc = inspect.getattr_static(cls, mname)
            if isinstance(desc, staticmethod):
                return "staticmethod"
            if isinstance(desc, classmethod):
                return "classmethod"
            if isinstance(desc, (types.FunctionType, property)):
                return "function"
        except Exception:
            pass
        return "unknown"

    # Unit / Integration tests
    for pt in path_tests:
        if not pt["feasible"]:
            continue
        called = False
        for node in pt["path"]:
            if "::" not in node:
                continue
            owner = node.split("::", 1)[0]

            # Regular functions 
            if owner in module and isinstance(module[owner], types.FunctionType):
                func = module[owner]
                try:
                    sig = inspect.signature(func)
                except Exception:
                    sig = None
                call_kwargs = {}
                if sig:
                    for p in sig.parameters.keys():
                        call_kwargs[p] = pt["inputs"].get(p, 0)
                try:
                    res, exc, lines = run_function_with_trace(func, call_kwargs)
                except Exception as e:
                    res = None; exc = e; lines = set()
                executed_lines_total.update(lines)
                test_id += 1
                rec = {
                    "test_id": test_id, "category": "unit", "target": owner,
                    "inputs": pt["inputs"], "output": _safe_repr(res),
                    "exception": repr(exc) if exc else None,
                    "covered_lines_sample": sorted(list(lines))[:10]
                }
                all_tests.append(rec); unit_tests.append(rec)
                called = True
                break

            # Class methods like Class.method
            if "." in owner:
                clsname, mname = owner.split('.', 1)
                cls = module.get(clsname)
                if isinstance(cls, type) and hasattr(cls, mname):
                    kind = _method_kind(cls, mname)
                    try:
                        if kind == "function":
                            # instance method
                            try:
                                instance = cls()
                                bound_method = getattr(instance, mname)
                            except Exception as e:
                                print(f"[!] Failed to instantiate {clsname}: {e}")
                                continue
                        elif kind == "staticmethod":
                            bound_method = getattr(cls, mname)
                        elif kind == "classmethod":
                            bound_method = getattr(cls, mname)
                        else:
                            # fallback: try to get attribute
                            try:
                                instance = cls()
                                bound_method = getattr(instance, mname)
                            except Exception:
                                bound_method = getattr(cls, mname)
                    except Exception as e:
                        print(f"[!] Error binding method {owner}: {e}")
                        continue

                    try:
                        sig = inspect.signature(bound_method)
                    except Exception:
                        sig = None
                    call_kwargs = {}
                    if sig:
                        for pname in sig.parameters.keys():
                            if pname in ("self", "cls"):
                                continue
                            call_kwargs[pname] = pt["inputs"].get(pname, 0)
                    try:
                        res, exc, lines = run_function_with_trace(bound_method, call_kwargs)
                    except Exception as e:
                        res = None; exc = e; lines = set()
                    executed_lines_total.update(lines)
                    test_id += 1
                    rec = {
                        "test_id": test_id, "category": "unit", "target": owner,
                        "inputs": pt["inputs"], "output": _safe_repr(res),
                        "exception": repr(exc) if exc else None,
                        "covered_lines_sample": sorted(list(lines))[:10]
                    }
                    all_tests.append(rec); unit_tests.append(rec)
                    called = True
                    break

        if not called:
            # Integration test fallback 
            triggered = False
            for fname, fn in module.items():
                if not isinstance(fn, types.FunctionType):
                    continue
                try:
                    sig = inspect.signature(fn)
                except Exception:
                    sig = None
                call_kwargs = {}
                if sig:
                    for p in sig.parameters.keys():
                        call_kwargs[p] = pt["inputs"].get(p, 0)
                try:
                    res, exc, lines = run_function_with_trace(fn, call_kwargs)
                    executed_lines_total.update(lines)
                    test_id += 1
                    rec = {
                        "test_id": test_id, "category": "integration", "target": fname,
                        "inputs": pt["inputs"], "output": _safe_repr(res),
                        "exception": repr(exc) if exc else None,
                        "covered_lines_sample": sorted(list(lines))[:10]
                    }
                    all_tests.append(rec); integration_tests.append(rec)
                    triggered = True
                    break
                except Exception:
                    continue
            if not triggered:
                continue

    # System testing 
    if "main" in module and isinstance(module["main"], types.FunctionType):
        main = module["main"]
        try:
            params = list(inspect.signature(main).parameters.keys())
        except Exception:
            params = []
        sample_map = {p: [0] for p in params}
        combos = list(itertools.product(*[sample_map[p] for p in params]))[:DEFAULT_SAMPLE_PER_PARAM] if params else [()]
        for combo in combos:
            kwargs = dict(zip(params, combo)) if params else {}
            try:
                res, exc, lines = run_function_with_trace(main, kwargs)
            except Exception as e:
                res = None; exc = e; lines = set()
            executed_lines_total.update(lines)
            test_id += 1
            rec = {
                "test_id": test_id, "category": "system", "target": "main",
                "inputs": kwargs, "output": _safe_repr(res),
                "exception": repr(exc) if exc else None,
                "covered_lines_sample": sorted(list(lines))[:10]
            }
            all_tests.append(rec); system_tests.append(rec)
    else:
        argv_samples = [[], ["--help"], ["input.txt"], ["--verbose", "input.txt"]]
        for argv in argv_samples:
            out, err, exc, lines = run_module_as_script_with_trace(path, argv)
            executed_lines_total.update(lines)
            test_id += 1
            rec = {
                "test_id": test_id, "category": "system", "target": "run_module",
                "inputs": argv, "output": _safe_repr(out),
                "exception": repr(exc) if exc else None,
                "covered_lines_sample": sorted(list(lines))[:10]
            }
            all_tests.append(rec); system_tests.append(rec)

    # Coverage computation and reports 
    covered_nodes, covered_edges = infer_covered_nodes_edges(executed_lines_total, func_cfgs, icfg)
    metrics = compute_metrics(icfg, covered_nodes, covered_edges, len([p for p in path_tests if p.get("feasible", True)]))
    metrics["executed_tests"] = len(all_tests)

    out_all = os.path.join(outdir, "all_tests.csv")
    with open(out_all, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test_id", "category", "target", "inputs", "output", "exception", "covered_lines_sample"])
        for r in all_tests:
            w.writerow([r["test_id"], r["category"], r["target"], _safe_repr(r["inputs"]),
                        r["output"], r["exception"], json.dumps(r["covered_lines_sample"])])

    def write_subset(filename, rows):
        filepath = os.path.join(outdir, filename)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["test_id", "target", "inputs", "output", "exception", "covered_lines_sample"])
            for r in rows:
                w.writerow([r["test_id"], r["target"], _safe_repr(r["inputs"]),
                            r["output"], r["exception"], json.dumps(r["covered_lines_sample"])])

    write_subset("unit_tests.csv", unit_tests)
    write_subset("integration_tests.csv", integration_tests)
    write_subset("system_tests.csv", system_tests)

    report = {"metrics": metrics, "num_path_tests": len(path_tests),
              "num_feasible_paths": len([p for p in path_tests if p.get("feasible", True)])}
    with open(os.path.join(outdir, "coverage_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nCoverage metrics:", json.dumps(metrics, indent=2))
    print(f"Outputs written to: {outdir}")
    # Extra visualizations 
    visualize_function_cfgs(func_cfgs)
    call_graph = build_function_call_graph(func_ast_map)
    visualize_function_call_graph(call_graph)

    return report


# directory processing 
def analyze_directory(dirpath):
    summary_rows = []
    for root,_,files in os.walk(dirpath):
        for file in files:
            if not file.endswith(".py"): continue
            filepath = os.path.join(root, file)
            print(f"\n=== Processing {filepath}")
            try:
                report = analyze_and_test_file(filepath)
                metrics = report.get("metrics", {})
                summary_rows.append({"file": filepath, **metrics})
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                traceback.print_exc()
    # write aggregate CSV
    outdir = ensure_outdir(OUT_DIR)
    aggfile = os.path.join(outdir, "aggregate_coverage.csv")
    with open(aggfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file","total_nodes","covered_nodes","node_pct","total_edges","covered_edges","edge_pct","total_paths"])
        for r in summary_rows:
            w.writerow([r.get("file"), r.get("total_nodes"), r.get("covered_nodes"), r.get("node_pct"), r.get("total_edges"), r.get("covered_edges"), r.get("edge_pct"), r.get("total_paths")])
    print(f"\nAggregate coverage written to {aggfile}")


def main():

    global OUT_DIR

    parser = argparse.ArgumentParser(description="Symbolic ICFG Tester (Z3 optional)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", help="Single python file to analyze")
    group.add_argument("--dir", "-d", help="Directory to analyze recursively")
    parser.add_argument("--out", "-o", help="Output directory (default ticfg_out)", default=OUT_DIR)
    args = parser.parse_args()

    if args.out:
        OUT_DIR = args.out
    ensure_outdir(OUT_DIR)

    if args.file:
        analyze_and_test_file(args.file)
    else:
        analyze_directory(args.dir)

if __name__ == "__main__":
    main()
