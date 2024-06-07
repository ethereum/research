from fields import S, M, B, ES, EM, EB
from stark import prove_low_degree, verify_low_degree
from fft import fft, inv_fft

INPUT_SIZE = 4096
coeffs = [M(3**i) for i in range(INPUT_SIZE)] + [M(0)] * INPUT_SIZE
evaluations = inv_fft(coeffs)
proof = prove_low_degree([EM(v) for v in evaluations])
length = (
    sum(32 * len(branch) for branch in sum(proof["branches"], []))
    + 32 * len(proof["roots"])
    + 32 * len(proof["leaf_values"])
    + 16 * len(proof["final_values"])
)
print("Proof length: {} bytes".format(length))

assert verify_low_degree(proof)
