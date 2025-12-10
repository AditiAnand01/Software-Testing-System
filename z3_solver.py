import ast

try:
    import z3
    HAS_Z3 = True
except Exception:
    HAS_Z3 = False

Z3_TIMEOUT_MS = 2000

# Z3 solving 
def try_solve_with_z3(conditions, param_names):
    if not HAS_Z3:
        return None
    try:
        s = z3.Solver()
        s.set("timeout", Z3_TIMEOUT_MS)
        zvars = {p: z3.Int(p) for p in param_names}

        for cond in conditions:
            try:
                # Parse condition to AST
                expr = ast.parse(cond, mode='eval').body
                zexpr = translate_expr_to_z3(expr, zvars)
                s.add(zexpr)
            except Exception as e:
                # Skip unsupported expressions gracefully
                print(f"[Z3-translate-warning] Skipping condition '{cond}': {e}")
                continue

        if s.check() == z3.sat:
            m = s.model()
            sol = {}
            for p in param_names:
                if p in zvars:
                    v = m[zvars[p]]
                    if v is not None:
                        try:
                            sol[p] = v.as_long()
                        except Exception:
                            try:
                                sol[p] = float(str(v))
                            except Exception:
                                sol[p] = 0
            return sol
        else:
            return None

    except Exception as e:
        print("[Z3-error]", e)
        return None
    

def translate_expr_to_z3(node, zvars):
    # Boolean combinations 
    if isinstance(node, ast.BoolOp):
        subexprs = [translate_expr_to_z3(v, zvars) for v in node.values]
        if isinstance(node.op, ast.And):
            return z3.And(*subexprs)
        elif isinstance(node.op, ast.Or):
            return z3.Or(*subexprs)
        else:
            raise ValueError(f"Unsupported BoolOp: {ast.dump(node.op)}")

    # Unary operators 
    elif isinstance(node, ast.UnaryOp):
        operand = translate_expr_to_z3(node.operand, zvars)
        if isinstance(node.op, ast.Not):
            return z3.Not(operand)
        elif isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return +operand
        else:
            raise ValueError(f"Unsupported UnaryOp: {ast.dump(node.op)}")

    # Binary arithmetic operators (with nesting) 
    elif isinstance(node, ast.BinOp):
        left = translate_expr_to_z3(node.left, zvars)
        right = translate_expr_to_z3(node.right, zvars)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            # Use real division for floats
            return left / right
        elif isinstance(node.op, ast.FloorDiv):
            return left // right
        elif isinstance(node.op, ast.Mod):
            return left % right
        elif isinstance(node.op, ast.Pow):
            # Z3 has no native power; use If for small integer powers
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                n = node.right.value
                expr = left
                for _ in range(n - 1):
                    expr *= left
                return expr
            else:
                return left ** right
        else:
            raise ValueError(f"Unsupported BinOp: {ast.dump(node.op)}")

    # Comparisons (chain supported)
    elif isinstance(node, ast.Compare):
        left = translate_expr_to_z3(node.left, zvars)
        zconds = []
        for op, right_node in zip(node.ops, node.comparators):
            right = translate_expr_to_z3(right_node, zvars)
            if isinstance(op, ast.Gt): zconds.append(left > right)
            elif isinstance(op, ast.GtE): zconds.append(left >= right)
            elif isinstance(op, ast.Lt): zconds.append(left < right)
            elif isinstance(op, ast.LtE): zconds.append(left <= right)
            elif isinstance(op, ast.Eq): zconds.append(left == right)
            elif isinstance(op, ast.NotEq): zconds.append(left != right)
            else:
                raise ValueError(f"Unsupported CompareOp: {ast.dump(op)}")
            left = right
        return z3.And(*zconds)

    # Name 
    elif isinstance(node, ast.Name):
        # default Int; could infer Bool if context known
        if node.id not in zvars:
            zvars[node.id] = z3.Int(node.id)
        return zvars[node.id]

    # Constant 
    elif isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, bool):
            return z3.BoolVal(val)
        elif isinstance(val, (int, float)):
            return val
        else:
            raise ValueError(f"Unsupported Constant type: {type(val)}")

    # Function calls (limited safe subset) 
    elif isinstance(node, ast.Call):
        fname = getattr(node.func, 'id', None)
        args = [translate_expr_to_z3(a, zvars) for a in node.args]
        if fname == 'abs':
            a = args[0]
            return z3.If(a >= 0, a, -a)
        elif fname == 'max':
            return z3.If(args[0] > args[1], args[0], args[1])
        elif fname == 'min':
            return z3.If(args[0] < args[1], args[0], args[1])
        else:
            raise ValueError(f"Unsupported function call: {fname}")

    # IfExp (ternary expression)
    elif isinstance(node, ast.IfExp):
        cond = translate_expr_to_z3(node.test, zvars)
        body = translate_expr_to_z3(node.body, zvars)
        orelse = translate_expr_to_z3(node.orelse, zvars)
        return z3.If(cond, body, orelse)

    else:
        raise ValueError(f"Unsupported expression type: {ast.dump(node)}")
    
