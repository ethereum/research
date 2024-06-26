from utils import (
    np, modinv, M31, log2, arange, array, zeros, append, m31_arith,
    mk_junk_data, device
)

from arithmetization_builder import (
    example, example_args, generate_constants_table, generate_arguments_table,
    generate_filled_trace, generate_next_state_function,
    get_public_args_indices
)

NUM_HASHES = 151

round_constants = mk_junk_data(1536).reshape((64, 24))

mds44 = array([
    [5, 7, 1, 3],
    [4, 6, 1, 1],
    [1, 3, 5, 7],
    [1, 1, 4, 6]
])

mds = zeros((24, 24))
for i in range(0, 24, 4):
    for j in range(0, 24, 4):
        mds[i:i+4, j:j+4] = mds44 * (2 if i==j else 1)

def poseidon_hash(in1, in2):
    state = zeros(in1.shape[:-1]+(24,))
    state[...,:8] = in1
    state[...,8:16] = in2
    for i in range(64):
        #print('Round {}: {}'.format(i*4, [int(x) for x in state[:3]]))
        state = (state + round_constants[i]) % M31
        if i >= 4 and i < 60:
            x = np.copy(state[...,0])
            state[...,0] = (x * x) % M31
            #print('Round {}: {}'.format(i*4+1, [int(x) for x in state[:3]]))
            state[...,0] = (state[...,0] * state[...,0]) % M31
            #print('Round {}: {}'.format(i*4+2, [int(x) for x in state[:3]]))
            state[...,0] = (state[...,0] * x) % M31
            #print('Round {}: {}'.format(i*4+3, [int(x) for x in state[:3]]))
        else:
            x = state
            state = (x * x) % M31
            #print('Round {}: {}'.format(i*4+1, [int(x) for x in state[:3]]))
            state = (state * state) % M31
            #print('Round {}: {}'.format(i*4+2, [int(x) for x in state[:3]]))
            state = (state * x) % M31
            #print('Round {}: {}'.format(i*4+3, [int(x) for x in state[:3]]))

        state = _matmul(state, mds) % M31
    return state[...,8:16]

def merkelize(data):
    assert len(data.shape) == 1
    output = zeros(data.shape[0] * 2)
    output[data.shape[0]:] = data
    for i in range(log2(data.shape[0])-1, 2, -1):
        hash_inputs = output[2**(i+1):2**(i+2)].reshape((-1,2,8))
        L, R = hash_inputs[...,0,:], hash_inputs[...,1,:]
        output[2**i: 2**(i+1)] = poseidon_hash(L, R).reshape((-1,))
    return output

# Handle with care. This is only safe if the values in at least one matrix
# are "small" (as floats are only reliable ints up to 2**53)
def _matmul(a, b):
    if hasattr(np, 'tensor'):
        return np.matmul(a.to(np.float64), b.to(np.float64)).to(np.int64)
    else:
        return np.matmul(a, b)
    #small = np.matmul((a&65535).to(np.float32), b.to(np.float32)).to(np.int64)
    #large = np.matmul((a>>16).to(np.float32), b.to(np.float32)).to(np.int64)
    #return small + (large << 16)

# 0: state[0...23] = state_prev[0...23] ** 2, state[24...47] = state_prev[0...23]
# 1: state[0...23] = state_prev[0...23] ** 2, state[24...47] = state_prev[24...47]
# 2: state[0...23] = state_prev[0...23] * state_prev[24...47], state_prev[24...47] = 0
# 3: state[0] = state_prev[0]**2, state[24] = state_prev[0]
# 4: state[0] = state_prev[0]**2, state[24] = state_prev[24]
# 5: state[0] = state_prev[0] * state_prev[24]
# 6: state[0...23] = MDS * state_prev[0...23], state_prev[24...47] = 0

# N(P(x) = P(xw))
# C(P(xw), P(x)) = N(P(x)) - P(xw) = 0

def poseidon_next_state(state, c, a, arith):
    one, add, mul = arith
    L = state[:24]
    R = state[24:]

    if c.ndim > 1 or np.count_nonzero(c[:8]) > 1:
        Z24 = np.zeros_like(R)
        Z8 = np.zeros_like(R[:8])
        Lp = (L + c[8:]) % M31
        MAT = _matmul(L.swapaxes(0, L.ndim-1), mds).swapaxes(0, L.ndim-1) % M31
        return (
            mul(c[0], append(mul(Lp, Lp), Lp)) +
            mul(c[1], append(mul(L, L), R)) +
            mul(c[2], append(mul(L, R), Z24)) +
            mul(c[3], append(mul(Lp[:1], Lp[:1]), Lp[1:], Lp)) +
            mul(c[4], append(mul(L[:1], L[:1]), L[1:24], R)) +
            mul(c[5], append(mul(L[:1], R[:1]), L[1:24], Z24)) +
            mul(c[6], append(MAT, Z24)) +
            mul(c[7], (
                mul((one - a[8]) % M31, append(a[:8], L[8:16], Z8, Z24)) +
                mul(a[8], append(L[8:16], a[:8], Z8, Z24))
            ) % M31)
        ) % M31
    else:
        if c[0]:
            Lp = (L + c[8:]) % M31
            return append(mul(Lp, Lp), Lp)
        elif c[1]:
            return append(mul(L, L), R)
        elif c[2]:
            return append(mul(L, R), zeros(24))
        elif c[3]:
            Lp = (L + c[8:]) % M31
            return append(mul(Lp[:1], Lp[:1]), Lp[1:], Lp)
        elif c[4]:
            return append(mul(L[:1], L[:1]), L[1:24], R)
        elif c[5]:
            return append(mul(L[:1], R[:1]), L[1:24], zeros(24))
        elif c[6]:
            MAT = _matmul(L, mds) % M31
            return append(MAT, zeros(24))
        elif c[7]:
            Z24 = np.zeros_like(R)
            Z8 = np.zeros_like(R[:8])
            return (
                mul((one - a[8]) % M31, append(a[:8], L[8:16], Z8, Z24)) +
                mul(a[8], append(L[8:16], a[:8], Z8, Z24))
            ) % M31
        else:
            return zeros(48)
    
OUTER = [
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,1,0]
]

INNER = [
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0]
]

INITIAL = [
    [0,0,0,0,0,0,0,1]
]

COMPONENT = INITIAL + (OUTER * 4 + INNER * 56 + OUTER * 4)


poseidon_constants = np.hstack((
    array(INITIAL + COMPONENT * NUM_HASHES),
    zeros((1 + 257 * NUM_HASHES, 24))
))
for i in range(NUM_HASHES):
    poseidon_constants[257*i+2:257*(i+1)+1:4, 8:] = round_constants

def arith_hash(in1, in2):
    state = zeros(48)
    arguments = zeros((258, 9))
    arguments[0, :8] = in1
    arguments[0, 8] = 1
    arguments[1, :8] = in2
    arguments[1, 8] = 1
    for i in range(258):
        #print('Round {}: {}'.format(i, [int(x) for x in state[:3]]))
        state = poseidon_next_state(
            state,
            poseidon_constants[i],
            arguments[i],
            m31_arith
        )
    return state[8:16]

def outer1(state, extra_constants, arguments, arith):
    one, add, mul = arith
    Lp = (state[:24] + extra_constants) % M31
    return append(mul(Lp, Lp), Lp)

def outer2(state, extra_constants, arguments, arith):
    one, add, mul = arith
    return append(mul(state[:24], state[:24]), state[24:])

def outer3(state, extra_constants, arguments, arith):
    one, add, mul = arith
    L = mul(state[:24], state[24:])
    return append(
        _matmul(L.swapaxes(0, state.ndim-1), mds)
        .swapaxes(0, state.ndim-1) % M31,
        np.zeros_like(L)
    )

def inner1(state, extra_constants, arguments, arith):
    one, add, mul = arith
    Lp = (state[:24] + extra_constants) % M31
    return append(mul(Lp[:1], Lp[:1]), Lp[1:], Lp)

def inner2(state, extra_constants, arguments, arith):
    one, add, mul = arith
    return append(mul(state[:1], state[:1]), state[1:])

def inner3(state, extra_constants, arguments, arith):
    one, add, mul = arith
    L = append(mul(state[:1], state[24:25]), state[1:24])
    return append(
        _matmul(L.swapaxes(0, state.ndim-1), mds)
        .swapaxes(0, state.ndim-1) % M31,
        np.zeros_like(L)
    )

def load_args(state, extra_constants, arguments, arith):
    one, add, mul = arith

    tag = arguments[8]
    Z32 = np.zeros_like(state[16:])
    return (
        mul((one - tag) % M31, append(arguments[:8], state[8:16], Z32)) +
        mul(tag, append(state[8:16], arguments[:8], Z32))
    ) % M31

outer = ["outer1", "outer2", "outer3"]
inner = ["inner1", "inner2", "inner3"]
component = ["load_args"] + outer * 4 + inner * 56 + outer * 4

round_constants = mk_junk_data(1536).reshape((64, 24))

poseidon_hasher = {
    "functions": {
        "outer1": outer1,
        "outer2": outer2,
        "outer3": outer3,
        "inner1": inner1,
        "inner2": inner2,
        "inner3": inner3,
        "load_args": load_args,
        "load_leaf": load_args,
    },
    "take_extra_constants": {"outer1": 24, "inner1": 24},
    "take_arguments": {"load_args": 9},
    "take_public_arguments": {"load_leaf": 9},
    "steps": ["load_leaf"] * 2 + component[1:],
    "trace_width": 48,
    "extra_constants": {
        "outer1": append(round_constants[:4], round_constants[-4:]),
        "inner1": round_constants[4:-4]
    }
}

poseidon_branch_hasher = {k:v for k,v in poseidon_hasher.items()}
poseidon_branch_hasher["steps"] = ["load_leaf"] * 2 + (component * NUM_HASHES)[1:]
poseidon_branch_hasher["extra_constants"]["outer1"] = append(*(
    (poseidon_branch_hasher["extra_constants"]["outer1"],) * NUM_HASHES
))
poseidon_branch_hasher["extra_constants"]["inner1"] = append(*(
    (poseidon_branch_hasher["extra_constants"]["inner1"],) * NUM_HASHES
))

def arith_hash2(in1, in2):
    constants = generate_constants_table(poseidon_hasher)
    arguments = generate_arguments_table(
        poseidon_hasher,
        {"load_leaf": [append(in1, array([1])), append(in2, array([1]))]}
    )
    prefilled_trace = generate_filled_trace(
        poseidon_hasher,
        constants,
        arguments
    )
    return prefilled_trace[len(poseidon_hasher["steps"]), 8:16]

def custom_trace_filler(arguments):
    trace_length = 2**(len(poseidon_branch_hasher["steps"])+1).bit_length()
    trace = np.zeros((
        trace_length,
        poseidon_branch_hasher["trace_width"]
    ), dtype=np.int64)
    trace[1, 8:16] = arguments["load_leaf"][0][:8]
    trace[2, :8] = trace[1, 8:16]
    trace[2, 8:16] = arguments["load_leaf"][1][:8]
    rc = np.array([[int(x) for x in z] for z in round_constants], dtype=np.int64)
    cpumds = np.array([[int(x) for x in z] for z in mds], dtype=np.int64)

    for i in range(NUM_HASHES):
        pos = 2 + 193 * i
        if i > 0:
            arg = arguments["load_args"][i-1]
            if arg[8] == 0:
                trace[pos, :8] = arg[:8]
                trace[pos, 8:16] = trace[pos-1, 8:16]
            else:
                trace[pos, :8] = trace[pos-1, 8:16]
                trace[pos, 8:16] = arg[:8]
        for r in range(4):
            Lp = (trace[pos, :24] + rc[r]) % M31
            trace[pos+1, :24] = Lp ** 2 % M31
            trace[pos+1, 24:] = Lp
            trace[pos+2, :24] = trace[pos+1, :24] ** 2 % M31
            trace[pos+2, 24:] = Lp
            L = trace[pos+2, :24] * Lp % M31
            trace[pos+3, :24] = _matmul(L, cpumds) % M31
            pos += 3
        for r in range(4, 60):
            Lp = (trace[pos, :24] + rc[r]) % M31
            trace[pos+1, :1] = Lp[:1] ** 2 % M31
            trace[pos+1, 1:24] = Lp[1:]
            trace[pos+1, 24:] = Lp
            trace[pos+2] = trace[pos+1]
            trace[pos+2, :1] = trace[pos+1, :1] ** 2 % M31
            L = append(
                trace[pos+2, :1] * trace[pos+2, 24:25] % M31,
                Lp[1:]
            )
            trace[pos+3, :24] = _matmul(L, cpumds) % M31
            pos += 3
        for r in range(60, 64):
            Lp = (trace[pos, :24] + rc[r]) % M31
            trace[pos+1, :24] = Lp ** 2 % M31
            trace[pos+1, 24:] = Lp
            trace[pos+2, :24] = trace[pos+1, :24] ** 2 % M31
            trace[pos+2, 24:] = Lp
            L = trace[pos+2, :24] * Lp % M31
            trace[pos+3, :24] = _matmul(L, cpumds) % M31
            pos += 3
    return trace.to(device)
