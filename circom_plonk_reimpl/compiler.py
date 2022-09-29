# A verification key generator for a simple zk language, reverse-engineered
# to match https://zkrepl.dev/ output

from utils import *

# Creates the inner-field representation of a given (section, index)
# Expects section = 1 for left, 2 for right, 3 for output
def S_position_to_f_inner(group_order, index, section):
    assert section in (1, 2, 3) and index < group_order
    return get_roots_of_unity(group_order)[index] * section

# Expects input in the form: [['a', 'b', 'c'], ...]
def make_s_polynomials(group_order, wires):
    if len(wires) > group_order:
        raise Exception("Group order too small")
    S = {
        1: [None] * group_order,
        2: [None] * group_order,
        3: [None] * group_order,
    }
    # For each variable, extract the list of (section, index) positions
    # where that variable is used
    variable_uses = {None: set()}
    for i, wire in enumerate(wires):
        for section, value in zip((1, 2, 3), wire):
            if value not in variable_uses:
                variable_uses[value] = set()
            variable_uses[value].add((i, section))
    for i in range(len(wires), group_order):
        for section in (1, 2, 3):
            variable_uses[None].add((i, section))
    # For each list of positions, rotate by one. For example, if some
    # variable is used in positions (1, 4), (1, 7) and (3, 2), then
    # we store:
    #
    # at S[1][7] the field element representing (1, 4)
    # at S[3][2] the field element representing (1, 7)
    # at S[1][4] the field element representing (3, 2)
    for _, uses in variable_uses.items():
        uses = sorted(uses)
        for i in range(len(uses)):
            next_i = (i+1) % len(uses)
            S[uses[next_i][1]][uses[next_i][0]] = S_position_to_f_inner(
                group_order, uses[i][0], uses[i][1]
            )
    return (S[1], S[2], S[3])

def is_valid_variable_name(name):
    return len(name) > 0 and name.isalnum() and name[0] not in '0123456789'

# Converts a arithmetic expression containing numbers, variables and {+, -, *}
# into a mapping of term to coefficient
#
# For example:
# ['a', '+', 'b', '*', 'c', '*', '5'] becomes {'a': 1, 'b*c': 5}
# 
# Note that this is a recursive algo, so the input can be a mix of tokens and
# mapping expressions
#
def simplify(exprs, first_is_negative=False):
    # Splits by + and - first, then *, to follow order of operations
    # The first_is_negative flag helps us correctly interpret expressions
    # like 6000 - 700 - 80 + 9 (that's 5229)
    if '+' in exprs:
        L = simplify(exprs[:exprs.index('+')], first_is_negative)
        R = simplify(exprs[exprs.index('+')+1:], False)
        return {
            x: L.get(x, 0) + R.get(x, 0) for x in set(L.keys()).union(R.keys())
        }
    elif '-' in exprs:
        L = simplify(exprs[:exprs.index('-')], first_is_negative)
        R = simplify(exprs[exprs.index('-')+1:], True)
        return {
            x: L.get(x, 0) + R.get(x, 0) for x in set(L.keys()).union(R.keys())
        }
    elif '*' in exprs:
        L = simplify(exprs[:exprs.index('*')], first_is_negative)
        R = simplify(exprs[exprs.index('*')+1:], first_is_negative)
        o = {}
        for k1 in L.keys():
            for k2 in R.keys():
                merged_key = min(k1,k2)+('*' if k1 and k2 else '')+max(k1,k2)
                o[merged_key] = L[k1] * R[k2]
        return o
    elif len(exprs) > 1:
        raise Exception("No ops, expected sub-expr to be a unit: {}"
                        .format(expr))
    elif exprs[0][0] == '-':
        return simplify([exprs[0][1:]], not first_is_negative)
    elif exprs[0].isnumeric():
        return {'': int(exprs[0]) * (-1 if first_is_negative else 1)}
    elif is_valid_variable_name(exprs[0]):
        return {exprs[0]: -1 if first_is_negative else 1}
    else:
        raise Exception("ok wtf is {}".format(exprs[0]))

# Converts an equation to a mapping of term to coefficient, and verifies that
# the operations in the equation are valid.
#
# Also outputs a triple containing the L and R input variables and the output
# variable
#
# Think of the list of (variable triples, coeffs) pairs as this language's
# version of "assembly" 
#
# Example valid equations, and output:
# a === 9                      ([None, None, 'a'], {'': 9})
# b <== a * c                  (['a', 'c', 'b'], {'a*c': 1})
# d <== a * c - 45 * a + 987   (['a', 'c', 'd'], {'a*c': 1, 'a': -45, '': 987})
#
# Example invalid equations:
# 7 === 7                      # Can't assign to non-variable
# a <== b * * c                # Two times signs in a row
# e <== a + b * c * d          # Multiplicative degree > 2
#
def eq_to_coeffs(eq):
    tokens = eq.split(' ')
    if tokens[1] in ('<==', '==='):
        # First token is the output variable
        out = tokens[0]
        # Convert the expression to coefficient map form
        coeffs = simplify(tokens[2:])
        # Handle the "-x === a * b" case
        if out[0] == '-':
            out = out[1:]
            coeffs['$output_coeff'] = -1
        # Check out variable name validity
        if not is_valid_variable_name(out):
            raise Exception("Invalid out variable name: {}".format(out))
        # Gather list of variables used in the expression
        variables = []
        for t in tokens[2:]:
            if is_valid_variable_name(t.lstrip('-')):
                variables.append(t.lstrip('-'))
        # Construct the list of allowed coefficients 
        allowed_coeffs = variables + ['', '$output_coeff']
        if len(variables) == 0:
            pass
        elif len(variables) == 1:
            allowed_coeffs.append(variables[0]+'*'+variables[0])
        elif len(variables) == 2:
            allowed_coeffs.append(min(variables)+'*'+max(variables))
        else:
            raise Exception("Max 2 variables, found {}".format(variables))
        # Check that only allowed coefficients are in the coefficient map
        for key in coeffs.keys():
            if key not in allowed_coeffs:
                raise Exception("Disallowed multiplication: {}".format(key))
        # Return output
        return variables + [None] * (2 - len(variables)) + [out], coeffs
    elif tokens[1] == 'public':
        return (
            [tokens[0], None, None],
            {tokens[0]: -1, '$output_coeff': 0, '$public': True}
        )
    else:
        raise Exception("Unsupported op: {}".format(tokens[1]))

# Generate the gate polynomials a list of 2-item tuples:
# Left: variable names, [in_L, in_R, out]
# Right: coeffs, {'': constant term, in_L: L term, in_R: R term,
#                 in_L*in_R: product term,
#                 '$output_coeff': coeff on output, 1 by default?}
def make_gate_polynomials(group_order, eqs):
    L = [f_inner(0) for _ in range(group_order)]
    R = [f_inner(0) for _ in range(group_order)]
    M = [f_inner(0) for _ in range(group_order)]
    O = [f_inner(0) for _ in range(group_order)]
    C = [f_inner(0) for _ in range(group_order)]
    for i, (variables, coeffs) in enumerate(eqs):
        L[i] = f_inner(-coeffs.get(variables[0], 0))
        R[i] = f_inner(-coeffs.get(variables[1], 0))
        C[i] = f_inner(-coeffs.get('', 0))
        O[i] = f_inner(coeffs.get('$output_coeff', 1))
        if None not in variables:
            merged_key = min(variables[:2])+'*'+max(variables[:2])
            M[i] = f_inner(-coeffs.get(merged_key, 0))
    return L, R, M, O, C

# Get the list of public variable assignments, in order
def get_public_assignments(coeffs):
    o = []
    no_more_allowed = False
    for coeff in coeffs:
        if coeff.get('$public', False) is True:
            if no_more_allowed:
                raise Exception("Public var declarations must be at the top")
            var_name = [x for x in list(coeff.keys()) if '$' not in x][0]
            if coeff != {'$public': True, '$output_coeff': 0, var_name: -1}:
                raise Exception("Malformatted coeffs: {}",format(coeffs))
            o.append(var_name)
        else:
            no_more_allowed = True
    return o

# Generate the verification key with the given setup, group order and equations
def make_verification_key(setup, group_order, eqs):
    if len(eqs) > group_order:
        raise Exception("Group order too small")
    # Convert equations into coeffs, eg.
    # 'c = a * b + 5' -> ['a', 'b', 'c'], {"a*b": 1, "": 5}
    eqs = [eq_to_coeffs(eq) if isinstance(eq, str) else eq for eq in eqs]
    L, R, M, O, C = make_gate_polynomials(group_order, eqs)
    S1, S2, S3 = make_s_polynomials(group_order, [v for (v, c) in eqs])
    return {
        "Qm": evaluations_to_point(setup, group_order, M),
        "Ql": evaluations_to_point(setup, group_order, L),
        "Qr": evaluations_to_point(setup, group_order, R),
        "Qo": evaluations_to_point(setup, group_order, O),
        "Qc": evaluations_to_point(setup, group_order, C),
        "S1": evaluations_to_point(setup, group_order, S1),
        "S2": evaluations_to_point(setup, group_order, S2),
        "S3": evaluations_to_point(setup, group_order, S3),
        "X_2": setup.X2,
        "w": get_root_of_unity(group_order)
    }
