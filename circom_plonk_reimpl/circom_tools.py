# A verification key generator for a simple zk language, reverse-engineered
# to match https://zkrepl.dev/ output

import py_ecc.bn128 as b
from fft import fft
f = b.FQ
f2 = b.FQ2

primitive_root = 5

def get_root_of_unity(group_order):
    return pow(primitive_root, (b.curve_order - 1) // group_order, b.curve_order)

SETUP_FILE_G1_STARTPOS = 80
SETUP_FILE_POWERS_POS = 60

class Setup(object):

    def __init__(self, G1_side, X2):
        self.G1_side = G1_side
        self.X2 = X2

    @classmethod
    def from_file(cls, filename):
        contents = open(filename, 'rb').read()
        # Byte 60 gives you the base-2 log of how many powers there are
        powers = 2**contents[SETUP_FILE_POWERS_POS]
        # Extract G1 points, which start at byte 80
        values = [
            int.from_bytes(contents[i: i+32], 'little')
            for i in range(SETUP_FILE_G1_STARTPOS,
                           SETUP_FILE_G1_STARTPOS + 32 * powers * 2, 32)
        ]
        assert max(values) < b.field_modulus
        # The points are encoded in a weird encoding, where all x and y points
        # are multiplied by a factor (for montgomery optimization?). We can extract
        # the factor because we know that the first point is the generator.
        factor = f(values[0]) / b.G1[0]
        values = [f(x) / factor for x in values]
        G1_side = [(values[i*2], values[i*2+1]) for i in range(powers)]
        print("Extracted G1 side, X^1 point: {}".format(G1_side[1]))
        # Search for start of G2 points. We again know that the first point is
        # the generator.
        pos = SETUP_FILE_G1_STARTPOS + 32 * powers * 2
        target = (factor * b.G2[0].coeffs[0]).n
        while pos < len(contents):
            v = int.from_bytes(contents[pos: pos+32], 'little')
            if v == target:
                break
            pos += 1
        print("Detected start of G2 side at byte {}".format(pos))
        X2_encoding = contents[pos + 32 * 4: pos + 32 * 8]
        X2_values = [
            f(int.from_bytes(X2_encoding[i: i + 32], 'little')) / factor
            for i in range(0, 128, 32)
        ]
        X2 = (f2(X2_values[:2]), f2(X2_values[2:]))
        assert b.is_on_curve(X2, b.b2)
        print("Extracted G2 side, X^1 point: {}".format(X2))
        assert b.pairing(b.G2, G1_side[1]) == b.pairing(X2, b.G1)
        print("X^1 points checked consistent")
        return cls(G1_side, X2)

# Encodes the KZG commitment to the given polynomial coeffs
def powers_to_point(setup, powers):
    if len(powers) > len(setup.G1_side):
        raise Exception("Not enough powers in setup")
    o = b.Z1
    for x, y in zip(powers, setup.G1_side):
        o = b.add(o, b.multiply(y, x % b.curve_order))
    return o

# Encodes the KZG commitment that evaluates to the given values in the group
def evaluations_to_point(setup, group_order, vals):
    powers = fft(vals, b.curve_order, get_root_of_unity(group_order), inv=True)
    return powers_to_point(setup, powers)

# Extracts a point from JSON in circom format
def interpret_json_point(p):
    if len(p) == 3 and isinstance(p[0], str) and p[2] == "1":
        return (f(int(p[0])), f(int(p[1])))
    elif len(p) == 3 and p == ["0", "1", "0"]:
        return b.Z1
    elif len(p) == 3 and isinstance(p[0], list) and p[2] == ["1", "0"]:
        return (
            f2([int(p[0][0]), int(p[0][1])]),
            f2([int(p[1][0]), int(p[1][1])]),
        )
    elif len(p) == 3 and p == [["0", "0"], ["1", "0"], ["0", "0"]]:
        return b.Z2
    raise Exception("cannot interpret that point: {}".format(p))

# Creates the inner-field representation of a given (section, index)
# Expects section = 1 for left, 2 for right, 3 for output
def S_position_to_field(group_order, index, section):
    assert section in (1, 2, 3) and index < group_order
    return (
        pow(get_root_of_unity(group_order), index, b.curve_order) * section
    ) % b.curve_order

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
            S[uses[next_i][1]][uses[next_i][0]] = S_position_to_field(
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
                merged_key = min(k1,k2) + ('*' if k1 and k2 else '') + max(k1,k2)
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
            coeffs['$flip_output'] = True
        # Check out variable name validity
        if not is_valid_variable_name(out):
            raise Exception("Invalid out variable name: {}".format(out))
        # Gather list of variables used in the expression
        variables = []
        for t in tokens[2:]:
            if is_valid_variable_name(t.lstrip('-')):
                variables.append(t.lstrip('-'))
        # Construct the list of allowed coefficients 
        allowed_coeffs = variables + ['', '$flip_output']
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
    else:
        raise Exception("Unsupported op: {}".format(tokens[1]))

# Generate the gate polynomials a list of 2-item tuples:
# Left: variable names, [in_L, in_R, out]
# Right: coeffs, {'': constant term, in_L: L term, in_R: R term,
#                 in_L*in_R: product term, '$flip_output': flip output to neg?}
def make_gate_polynomials(group_order, eqs):
    L = [0] * group_order
    R = [0] * group_order
    M = [0] * group_order
    O = [0] * group_order
    C = [0] * group_order
    for i, (variables, coeffs) in enumerate(eqs):
        L[i] = -coeffs.get(variables[0], 0)
        R[i] = -coeffs.get(variables[1], 0)
        C[i] = -coeffs.get('', 0)
        O[i] = (-1 if '$flip_output' in coeffs else 1)
        if None not in variables:
            M[i] = -coeffs.get(min(variables[:2])+'*'+max(variables[:2]), 0)
    return L, R, M, O, C        

# Generate the verification key with the given setup, group order and equations
def make_verification_key(setup, group_order, eqs):
    if len(eqs) > group_order:
        raise Exception("Group order too small")
    # Convert equations into coeffs, eg.
    # 'c = a * b + 5' -> ['a', 'b', 'c'], {"a*b": 1, "": 5}
    eqs = [eq_to_coeffs(eq) if isinstance(eq, str) else eq for eq in eqs]
    variable_uses = [variables for (variables, coeffs) in eqs]
    L, R, M, O, C = make_gate_polynomials(group_order, eqs)
    S1, S2, S3 = make_s_polynomials(group_order, variable_uses)
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
