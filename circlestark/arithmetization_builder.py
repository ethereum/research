from utils import (
    np, modinv, M31, log2, arange, array, zeros, append, m31_arith,
    mk_junk_data
)

# We create our own mini-DSL that lets you specify a program as a series of
# opcodes. This then converts it into more efficient functions for filling
# the trace

# Here is a basic example of a program in that mini-DSL
def example_load_args(state, constants, arguments, arith):
    one, add, mul = arith
    return add(state, arguments)

def example_step(state, constants, arguments, arith):
    one, add, mul = arith
    return np.stack((
        add(mul(state[1], state[2]), one),
        add(mul(state[2], state[0]), one),
        add(mul(state[0], state[1]), one),
    ))

example = {
    "functions": {
        "example_load_args": example_load_args,
        "example_step": example_step
    },
    "take_extra_constants": {},
    "take_arguments": {},
    "take_public_arguments": {"example_load_args": 3},
    "steps": ["example_load_args"]+["example_step"]*98+["example_load_args"],
    "trace_width": 3,
    "extra_constants": {},
}
example_args = {
    "example_load_args": [[3,0,0], [0,0,0]]
}

# Given an object in the format above, generate the constants table. This
# has two parts: k columns for k different opcodes, and then opcode-specific
# constants
def generate_constants_table(obj):
    functions_list = list(obj["functions"].keys())
    extra_constants_width = max(obj["take_extra_constants"].values() or (0,))
    base_width = len(functions_list)
    constants = zeros((
        len(obj["steps"]),
        base_width + extra_constants_width
    ))
    constants_indices = {k:0 for k in obj["take_extra_constants"]}
    for i, function in enumerate(obj["steps"]):
        index = functions_list.index(function)
        constants[i, index] = 1
        if function in obj["take_extra_constants"]:
            width = obj["take_extra_constants"][function]
            constants[i, base_width:base_width + width] = array(
                obj["extra_constants"][function][constants_indices[function]]
            )
            constants_indices[function] += 1
    return constants

# What is the width of the arguments polynomial?
def get_arguments_width(obj):
    take_args = {**obj["take_arguments"], **obj["take_public_arguments"]}
    return max(take_args.values() or (0,))

# Generate the arguments polynomial
def generate_arguments_table(obj, arguments):
    take_args = {**obj["take_arguments"], **obj["take_public_arguments"]}
    arguments_width = max(take_args.values() or (0,))
    o = zeros((
        len(obj["steps"]),
        arguments_width
    ))
    arguments_indices = {k:0 for k in take_args}
    for i, function in enumerate(obj["steps"]):
        if function in take_args:
            width = take_args[function]
            o[i, :width] = array(
                arguments[function][arguments_indices[function]]
            )
            arguments_indices[function] += 1
    return o

# Fill the trace
def generate_filled_trace(obj, constants, arguments):
    trace_length = 2**(len(obj["steps"])+1).bit_length()
    trace = zeros((
        trace_length,
        obj["trace_width"]
    ))
    function_count = len(list(obj["functions"].keys()))
    functions = obj["functions"]
    for i, step in enumerate(obj["steps"]):
        f = functions[step]
        trace[i+1] = f(
            trace[i],
            constants[i][function_count:],
            arguments[i],
            m31_arith
        )
    return trace

# This is the next_state_vector function that needs to be passed in to
# mk_stark and verify_stark
def generate_next_state_function(obj):

    function_count = len(list(obj["functions"].keys()))

    def next_state(state, constants, arguments, arith):
        one, add, mul = arith
        o = np.zeros_like(state)
        c = constants[function_count:]
        for i, function in enumerate(list(obj["functions"].values())):
            o += mul(
                constants[i],
                function(state, c, arguments, arith)
            )
        return o % M31

    return next_state

# Get the positions of the public arguments in the arguments polynomial
def get_public_args_indices(obj):
    return array([
        i for i,v in enumerate(obj["steps"])
        if v in obj["take_public_arguments"]
    ])
