import ast
import networkx as nx
from collections import defaultdict
from matplotlib import pyplot as plt

from utils import safe_unparse


# Config 
MAX_PATHS = 800
LOOP_UNROLL_DEPTH = 2
Z3_TIMEOUT_MS = 2000
DEFAULT_INT_RANGE = (-3, 3)
DEFAULT_FLOAT_RANGE = (-3.0, 3.0)
DEFAULT_SAMPLE_PER_PARAM = 4

OUT_DIR = "ticfg_out"  # output CSVs/JSON saved here

# CFG builder 
class CFGBuilder:
    def __init__(self, owner, loop_unroll=LOOP_UNROLL_DEPTH):
        self.owner = owner
        self.graph = nx.DiGraph()
        self.counter = 0
        self.prev_nodes = []
        self.loop_unroll = loop_unroll

    def new_node(self, label, astnode=None):
        nid = f"{self.owner}::{self.counter}"
        lineno = getattr(astnode, "lineno", None) if astnode is not None else None
        self.graph.add_node(nid, label=label, lineno=lineno, astnode=astnode)
        self.counter += 1
        return nid

    def add_edge_from_prev(self, nid):
        for p in self.prev_nodes:
            self.graph.add_edge(p, nid)

    def build_from_body(self, body):
        for stmt in body:
            if isinstance(stmt, ast.If):
                cond_label = f"If: {safe_unparse(stmt.test)}"
                cond = self.new_node(cond_label, stmt.test)
                self.add_edge_from_prev(cond)
                # true branch
                self.prev_nodes = [cond]
                self.build_from_body(stmt.body)
                true_tails = self.prev_nodes or [cond]
                # false branch
                self.prev_nodes = [cond]
                self.build_from_body(stmt.orelse)
                false_tails = self.prev_nodes or [cond]
                # merge
                merge = self.new_node("MERGE", stmt)
                for t in true_tails:
                    self.graph.add_edge(t, merge)
                for f in false_tails:
                    self.graph.add_edge(f, merge)
                self.prev_nodes = [merge]

            elif isinstance(stmt, (ast.For, ast.While)):
                if isinstance(stmt, ast.While):
                    cond_expr = getattr(stmt, "test", None)
                    label = f"LoopCond: {safe_unparse(cond_expr)}"
                else:
                    label = f"ForLoop: {safe_unparse(stmt.target)} in {safe_unparse(stmt.iter)}"
                cond = self.new_node(label, stmt)
                self.add_edge_from_prev(cond)
                tail_nodes = [cond]
                for _ in range(self.loop_unroll):
                    self.prev_nodes = tail_nodes.copy()
                    self.build_from_body(list(stmt.body))
                    tail_nodes = self.prev_nodes.copy() or tail_nodes
                exit_node = self.new_node("LoopExit", stmt)
                for t in tail_nodes:
                    self.graph.add_edge(t, exit_node)
                self.prev_nodes = [exit_node]

            elif isinstance(stmt, ast.Try):
                try_node = self.new_node("TRY", stmt)
                self.add_edge_from_prev(try_node)
                self.prev_nodes = [try_node]
                self.build_from_body(stmt.body)
                try_tails = self.prev_nodes or [try_node]
                except_tails = []
                for handler in stmt.handlers:
                    hnode = self.new_node(f"EXCEPT {safe_unparse(handler.type) if handler.type else 'Exception'}", handler)
                    self.graph.add_edge(try_node, hnode)
                    self.prev_nodes = [hnode]
                    self.build_from_body(handler.body)
                    except_tails.extend(self.prev_nodes or [hnode])
                if stmt.finalbody:
                    fin = self.new_node("FINALLY", stmt)
                    for t in try_tails + except_tails:
                        self.graph.add_edge(t, fin)
                    self.prev_nodes = [fin]
                    self.build_from_body(stmt.finalbody)
                else:
                    merge = self.new_node("TRY_MERGE", stmt)
                    for t in try_tails + except_tails:
                        self.graph.add_edge(t, merge)
                    self.prev_nodes = [merge]

            elif isinstance(stmt, ast.Return):
                rn = self.new_node(f"Return: {safe_unparse(stmt.value) if stmt.value else ''}", stmt)
                self.add_edge_from_prev(rn)
                self.prev_nodes = []

            elif isinstance(stmt, ast.Assign):
                an = self.new_node(f"Assign: {safe_unparse(stmt)}", stmt)
                self.add_edge_from_prev(an)
                self.prev_nodes = [an]

            elif isinstance(stmt, ast.Expr):
                en = self.new_node(f"Expr: {safe_unparse(stmt.value)}", stmt)
                self.add_edge_from_prev(en)
                self.prev_nodes = [en]

            else:
                gn = self.new_node(f"{type(stmt).__name__}: {safe_unparse(stmt)}", stmt)
                self.add_edge_from_prev(gn)
                self.prev_nodes = [gn]

# build CFGs from all functions in source code
def build_all_cfgs(source_code, loop_unroll=LOOP_UNROLL_DEPTH):
    tree = ast.parse(source_code)
    func_cfgs = {}
    func_entries = {}
    func_ast_map = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            owner = node.name
            b = CFGBuilder(owner, loop_unroll=loop_unroll)
            entry = b.new_node(f"ENTRY:{owner}", node)
            b.prev_nodes = [entry]
            b.build_from_body(node.body)
            exit_node = b.new_node(f"EXIT:{owner}", node)
            if b.prev_nodes:
                for p in b.prev_nodes:
                    b.graph.add_edge(p, exit_node)
            func_cfgs[owner] = b.graph
            func_entries[owner] = entry
            func_ast_map[owner] = node
        elif isinstance(node, ast.ClassDef):
            clsname = node.name
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    owner = f"{clsname}.{item.name}"
                    b = CFGBuilder(owner, loop_unroll=loop_unroll)
                    entry = b.new_node(f"ENTRY:{owner}", item)
                    b.prev_nodes = [entry]
                    b.build_from_body(item.body)
                    exit_node = b.new_node(f"EXIT:{owner}", item)
                    if b.prev_nodes:
                        for p in b.prev_nodes:
                            b.graph.add_edge(p, exit_node)
                    func_cfgs[owner] = b.graph
                    func_entries[owner] = entry
                    func_ast_map[owner] = item
    return func_cfgs, func_entries, func_ast_map

# build ICFG from all function CFGs
def build_icfg(source_code, loop_unroll=LOOP_UNROLL_DEPTH):
    func_cfgs, func_entries, func_ast_map = build_all_cfgs(source_code, loop_unroll=loop_unroll)
    icfg = nx.DiGraph()
    for g in func_cfgs.values():
        icfg = nx.compose(icfg, g)
    # lineno map for each owner
    lineno_map = {}
    for owner,g in func_cfgs.items():
        mp = defaultdict(list)
        for n,d in g.nodes(data=True):
            ln = d.get("lineno")
            if ln is not None:
                mp[ln].append(n)
        lineno_map[owner] = mp
    # parse AST to link calls
    tree = ast.parse(source_code)
    func_defs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_defs[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    func_defs[f"{node.name}.{item.name}"] = item
    for owner,fnode in func_defs.items():
        for callsite in ast.walk(fnode):
            if isinstance(callsite, ast.Call):
                callee_name = None
                if isinstance(callsite.func, ast.Name):
                    callee_name = callsite.func.id
                elif isinstance(callsite.func, ast.Attribute):
                    callee_name = callsite.func.attr
                if not callee_name:
                    continue
                candidates = [k for k in func_entries.keys() if k.endswith(callee_name)]
                chosen = callee_name if callee_name in func_entries else (sorted(candidates, key=lambda x: len(x))[0] if candidates else None)
                if not chosen:
                    continue
                ln = getattr(callsite, "lineno", None)
                call_nodes = lineno_map.get(owner, {}).get(ln, [])
                if not call_nodes:
                    candn = []
                    for n,d in func_cfgs[owner].nodes(data=True):
                        nln = d.get("lineno")
                        if nln is not None and ln is not None and nln <= ln:
                            candn.append((nln, n))
                    if candn:
                        candn.sort()
                        call_nodes = [candn[-1][1]]
                entry = func_entries.get(chosen)
                if entry:
                    for cn in call_nodes:
                        icfg.add_edge(cn, entry)
    return icfg, func_cfgs, func_entries, func_ast_map

# enumerate paths in ICFG
def enumerate_icfg_paths(icfg, max_paths=MAX_PATHS):
    starts = [n for n in icfg.nodes() if icfg.in_degree(n) == 0]
    exits = [n for n in icfg.nodes() if icfg.out_degree(n) == 0]
    paths = []
    for s in starts:
        for e in exits:
            for p in nx.all_simple_paths(icfg, s, e, cutoff=200):
                paths.append(p)
                if len(paths) >= max_paths:
                    return paths
    return paths


# coverage & visualization 
# compute coverage metrics
def compute_metrics(icfg, covered_nodes, covered_edges, total_paths):
    total_nodes = icfg.number_of_nodes()
    total_edges = icfg.number_of_edges()
    node_pct = (len(covered_nodes)/total_nodes*100.0) if total_nodes else 100.0
    edge_pct = (len(covered_edges)/total_edges*100.0) if total_edges else 100.0
    return {"total_nodes": total_nodes, "covered_nodes": len(covered_nodes), "node_pct": round(node_pct,2),
            "total_edges": total_edges, "covered_edges": len(covered_edges), "edge_pct": round(edge_pct,2),
            "total_paths": total_paths}

# visualize ICFG with covered edges highlighted
def visualize_icfg(icfg, covered_edges):
    plt.figure(figsize=(12,8))
    pos = nx.spring_layout(icfg, seed=42)
    labels = nx.get_node_attributes(icfg,"label")
    nx.draw_networkx_nodes(icfg, pos, node_size=700, node_color="lightgray")
    nx.draw_networkx_labels(icfg, pos, labels, font_size=8)
    cov = set(covered_edges)
    other = [e for e in icfg.edges() if e not in cov]
    nx.draw_networkx_edges(icfg, pos, edgelist=other, edge_color="lightgray", arrows=True)
    if cov:
        nx.draw_networkx_edges(icfg, pos, edgelist=list(cov), edge_color="green", width=2.2, arrows=True)
    plt.title("ICFG (green = covered edges)")
    plt.axis("off")
    plt.show()
    plt.close()