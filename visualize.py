# --- Graph drawing utilities (drop into your script) ---
import os
import ast
import csv
import textwrap
import networkx as nx
import matplotlib.pyplot as plt

OUT_DIR = "ticfg_out" 
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

def _choose_layout(G, prog="dot", seed=42):
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog=prog)  # pygraphviz
        return pos
    except Exception:
        try:
            pos = nx.nx_pydot.graphviz_layout(G, prog=prog)  # pydot
            return pos
        except Exception:
            return nx.spring_layout(G, seed=seed)

def _format_node_label(n, data, max_width=60, show_id=True):
    src = None
    if isinstance(data, dict):
        for k in ("src", "code", "label", "text"):
            if k in data and data[k]:
                src = str(data[k]).strip()
                break
        lineno = data.get("lineno") or data.get("line") or data.get("lineno_from")
    else:
        src = str(data)
        lineno = None

    if src:
        src = src.replace("\n", " ").strip()
        if len(src) > max_width:
            src = src[: max_width - 3].rstrip() + "..."
        wrapped = "\n".join(textwrap.wrap(src, width=30))
        if show_id and lineno:
            return f"{n}\n(L{lineno}) {wrapped}"
        elif lineno:
            return f"(L{lineno}) {wrapped}"
        elif show_id:
            return f"{n}\n{wrapped}"
        else:
            return wrapped
    else:
        return str(n)

def _get_edge_label(u, v, data):
    if not isinstance(data, dict):
        return ""
    for k in ("label", "cond", "condition", "edge_label"):
        vlab = data.get(k)
        if vlab:
            return str(vlab)
    return ""

def extract_code(data):
    for key in ["src", "code", "line", "stmt", "text", "label"]:
        val = data.get(key)
        if val:
            return str(val).strip()
    return ""

def merge_duplicate_nodes(G):
    lineno_to_node = {}
    merged_G = nx.DiGraph()

    for n, data in G.nodes(data=True):
        lineno = data.get("lineno")
        src = extract_code(data)
        if not src.strip():
            continue

        if lineno is None:
            merged_G.add_node(n, **data)
            continue

        if lineno not in lineno_to_node:
            lineno_to_node[lineno] = n
            merged_G.add_node(n, **data)
        else:
            existing_node = lineno_to_node[lineno]
            existing_data = merged_G.nodes[existing_node]
            old_src = extract_code(existing_data)
            if src and src not in old_src:
                existing_data["src"] = (old_src + " ; " + src).strip(" ;")

    # copy edges
    for u, v, edata in G.edges(data=True):
        u_line = G.nodes[u].get("lineno", u)
        v_line = G.nodes[v].get("lineno", v)
        u_new = lineno_to_node.get(u_line, u)
        v_new = lineno_to_node.get(v_line, v)
        if u_new != v_new:
            merged_G.add_edge(u_new, v_new, **edata)

    return merged_G


def save_node_mapping_csv(G, out_csv):
    # Write node ID, line number, code, and outgoing edges with conditions.
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "line_number", "code", "outgoing_edges"])

        for n, data in G.nodes(data=True):
            lineno = data.get("lineno", "")
            code = extract_code(data)
            code = code.replace("\n", " ").strip()

            outgoing = []
            for _, v, edata in G.out_edges(n, data=True):
                cond = edata.get("label") or "move"
                tgt_lineno = G.nodes[v].get("lineno", "")
                outgoing.append(f"-> L{tgt_lineno} ({cond})")

            outgoing_text = "; ".join(outgoing) if outgoing else ""
            writer.writerow([n, lineno, code, outgoing_text])


def visualize_function_cfgs(func_cfgs, out_dir="graphs"):
    os.makedirs(out_dir, exist_ok=True)

    for func_name, G in func_cfgs.items():
        G_clean = merge_duplicate_nodes(G)

        for u, v, edata in G_clean.edges(data=True):
            if not edata.get("label"):
                edata["label"] = "move"

        try:
            pos = nx.nx_agraph.graphviz_layout(G_clean, prog="dot")
        except Exception:
            pos = nx.spring_layout(G_clean, seed=42)

        plt.figure(figsize=(13, 8))

        labels = {}
        for n, data in G_clean.nodes(data=True):
            lineno = data.get("lineno")
            src = extract_code(data)
            if len(src) > 80:
                src = src[:77] + "..."
            labels[n] = f"L{lineno}: {src}" if lineno else src or str(n)

        node_colors = []
        for _, data in G_clean.nodes(data=True):
            code = extract_code(data).lower()
            if any(k in code for k in ("if ", "elif ", "else")):
                node_colors.append("#ff9999")
            elif any(k in code for k in ("for ", "while ")):
                node_colors.append("#ffe680")
            elif "return" in code:
                node_colors.append("#b3ffb3")
            else:
                node_colors.append("#9fd3ff")

        nx.draw(
            G_clean, pos, with_labels=False,
            node_color=node_colors, node_size=3400,
            edgecolors="black", linewidths=1.0
        )

        for node, (x, y) in pos.items():
            plt.text(
                x, y, labels[node],
                ha='center', va='center', fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85)
            )

        edge_labels = nx.get_edge_attributes(G_clean, "label")
        nx.draw_networkx_edge_labels(G_clean, pos, edge_labels=edge_labels, font_size=8, font_color="darkgreen")

        plt.title(f"Control Flow Graph: {func_name}", fontsize=12, fontweight="bold")
        plt.axis("off")

        png_path = os.path.join(out_dir, f"cfg_{func_name}.png")
        csv_path = os.path.join(out_dir, f"cfg_{func_name}_mapping.csv")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
        save_node_mapping_csv(G_clean, csv_path)

        print(f"[+] Saved CFG for {func_name} -> {png_path}")
        print(f"[+] Mapping CSV -> {csv_path}")

def validate_graph_has_src(G, graph_name="graph"):
    # Print warnings for nodes missing source info (helpful when images show just numbers).
    missing = []
    for n, data in G.nodes(data=True):
        if not (data.get("src") or data.get("code") or data.get("label") or data.get("lineno") or data.get("line")):
            missing.append(n)
    if missing:
        print(f"[!] Warning: {len(missing)} nodes in {graph_name} lack 'src'/'label'/'lineno' metadata. "
              "Images may show numeric IDs only. Consider enriching func_cfgs/icfg nodes with 'src' or 'lineno'.")
        # show a sample
        print("    sample node ids without metadata:", missing[:10])

def build_function_call_graph(func_ast_map):
    """
    Build a function call graph (FCG) where nodes are functions and edges indicate calls.
    func_ast_map: {func_name: ast.FunctionDef or AST node}
    """
    call_graph = nx.DiGraph()
    # ensure all functions are present as nodes
    for func_name in func_ast_map.keys():
        call_graph.add_node(func_name)

    for func_name, func_node in func_ast_map.items():
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                # direct function name call
                if isinstance(node.func, ast.Name):
                    callee = node.func.id
                    if not call_graph.has_node(callee):
                        call_graph.add_node(callee)  # include external/simple functions too
                    # count multiple calls (increase a 'count' attribute)
                    if call_graph.has_edge(func_name, callee):
                        call_graph[func_name][callee]["count"] = call_graph[func_name][callee].get("count", 1) + 1
                    else:
                        call_graph.add_edge(func_name, callee, count=1)
                # method calls like obj.method()
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.attr, str):
                    callee = node.func.attr
                    if not call_graph.has_node(callee):
                        call_graph.add_node(callee)
                    if call_graph.has_edge(func_name, callee):
                        call_graph[func_name][callee]["count"] = call_graph[func_name][callee].get("count", 1) + 1
                    else:
                        call_graph.add_edge(func_name, callee, count=1)
    return call_graph

def visualize_function_call_graph(call_graph, out_dir=GRAPH_DIR):
    """
    Visualize and save the Function Call Graph (FCG).
    Edge labels show call counts; nodes are function names.
    """
    os.makedirs(out_dir, exist_ok=True)
    validate_graph_has_src(call_graph, graph_name="function_call_graph")  # might be empty but fine

    plt.figure(figsize=(10, 8))
    pos = _choose_layout(call_graph, prog="dot")

    # labels are function names (node ids)
    labels = {n: str(n) for n in call_graph.nodes()}
    nx.draw_networkx_nodes(call_graph, pos, node_color="#d9f2d9", node_size=2200, edgecolors="black")
    nx.draw_networkx_labels(call_graph, pos, labels=labels, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(call_graph, pos, arrows=True, arrowsize=14)

    # edge labels — show call counts if present
    edge_labels = {}
    for u, v, data in call_graph.edges(data=True):
        cnt = data.get("count")
        if cnt is not None:
            edge_labels[(u, v)] = f"calls: {cnt}"
        else:
            edge_labels[(u, v)] = data.get("label", "")

    edge_labels = {k: v for k, v in edge_labels.items() if v}
    if edge_labels:
        nx.draw_networkx_edge_labels(call_graph, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Function Call Graph", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "function_call_graph.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved Function Call Graph -> {out_path}")

    # save node mapping CSV (functions)
    map_csv = os.path.join(out_dir, "function_call_graph_node_mapping.csv")
    save_node_mapping_csv(call_graph, map_csv)

# --- Optional: improved visualize_icfg to include edge labels and node source ---
def visualize_icfg_with_code(icfg, out_dir=GRAPH_DIR, covered_edges=None):
    """
    Visualize ICFG with source snippets on nodes and conditions on edges.
    If covered_edges is provided, they will be highlighted.
    """
    os.makedirs(out_dir, exist_ok=True)
    G = icfg
    validate_graph_has_src(G, graph_name="ICFG")

    plt.figure(figsize=(14, 10))
    pos = _choose_layout(G, prog="dot")

    labels = {n: _format_node_label(n, data) for n, data in G.nodes(data=True)}
    node_colors = []
    for n, data in G.nodes(data=True):
        code = (data.get("src") or data.get("code") or data.get("label") or "").lower()
        if "if " in code or code.strip().startswith("if"):
            node_colors.append("#ffb3b3")
        elif "while" in code or "for " in code:
            node_colors.append("#ffe6a6")
        elif "return" in code:
            node_colors.append("#c6f5c6")
        else:
            node_colors.append("#cfe9ff")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2600, edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold")

    # draw edges with optional coverage highlighting
    if covered_edges:
        # draw covered edges thicker/green; others thin/gray
        covered_set = set(covered_edges)
        covered_edges_list = [e for e in G.edges() if e in covered_set]
        other_edges_list = [e for e in G.edges() if e not in covered_set]
        if other_edges_list:
            nx.draw_networkx_edges(G, pos, edgelist=other_edges_list, edge_color="#888", arrows=True, arrowsize=10)
        if covered_edges_list:
            nx.draw_networkx_edges(G, pos, edgelist=covered_edges_list, edge_color="#2ca02c", width=2.2, arrows=True, arrowsize=12)
    else:
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10)

    edge_labels = {(u, v): _get_edge_label(u, v, data) for u, v, data in G.edges(data=True)}
    edge_labels = {k: v for k, v in edge_labels.items() if v}
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title("Interprocedural CFG (ICFG)", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "icfg_full.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved ICFG -> {out_path}")

    # save node mapping CSV
    map_csv = os.path.join(out_dir, "icfg_node_mapping.csv")
    save_node_mapping_csv(G, map_csv)


