import compiler as c
import prover as p
import verifier as v
import json

def basic_test():
    setup = c.Setup.from_file('powersOfTau28_hez_final_11.ptau')
    print("Extracted setup")
    vk = c.make_verification_key(setup, 8, ['c <== a * b'])
    print("Generated verification key")
    circom_output = json.load(open('main.plonk.vkey.json'))
    for key in ('Qm', 'Ql', 'Qr', 'Qo', 'Qc', 'S1', 'S2', 'S3', 'X_2'):
        if c.interpret_json_point(circom_output[key]) != vk[key]:
            raise Exception("Mismatch {}: ours {} theirs {}"
                            .format(key, vk[key], circom_output[key]))
    assert vk['w'] == int(circom_output['w'])
    print("Basic test success")
    return setup

# Equivalent to this circom code:
#
# template Example () {
#    signal input a;
#    signal input b;
#    signal c;
#    c <== a * b + a;
# }
def ab_plus_a_test(setup):
    vk = c.make_verification_key(setup, 8, ['ab === a - c', '-ab === a * b'])
    print("Generated verification key")
    circom_output = json.load(open('main.plonk.vkey-58.json'))
    for key in ('Qm', 'Ql', 'Qr', 'Qo', 'Qc', 'S1', 'S2', 'S3', 'X_2'):
        if c.interpret_json_point(circom_output[key]) != vk[key]:
            raise Exception("Mismatch {}: ours {} theirs {}"
                            .format(key, vk[key], circom_output[key]))
    assert vk['w'] == int(circom_output['w'])
    print("ab+a test success")

def one_public_input_test(setup):
    vk = c.make_verification_key(setup, 8, ['c public', 'c === a * b'])
    print("Generated verification key")
    circom_output = json.load(open('main.plonk.vkey-59.json'))
    for key in ('Qm', 'Ql', 'Qr', 'Qo', 'Qc', 'S1', 'S2', 'S3', 'X_2'):
        if c.interpret_json_point(circom_output[key]) != vk[key]:
            raise Exception("Mismatch {}: ours {} theirs {}"
                            .format(key, vk[key], circom_output[key]))
    assert vk['w'] == int(circom_output['w'])
    print("One public input test success")

def prover_test(setup):
    print("Beginning prover test")
    eqs = ['e public', 'c <== a * b', 'e <== c * d']
    assignments = {'a': 3, 'b': 4, 'c': 12, 'd': 5, 'e': 60}
    return p.prove_from_witness(setup, 8, eqs, assignments)
    print("Prover test success")

def verifier_test(setup, proof):
    print("Beginning verifier test")
    eqs = ['e public', 'c <== a * b', 'e <== c * d']
    public = [60]
    vk = c.make_verification_key(setup, 8, eqs)
    assert v.verify_proof(setup, 8, vk, proof, public, optimized=False)
    assert v.verify_proof(setup, 8, vk, proof, public, optimized=True)
    print("Verifier test success")

def factorization_test(setup):
    print("Beginning test: prove you know small integers that multiply to 91")
    eqs = """
        n public
        pb0 === pb0 * pb0
        pb1 === pb1 * pb1
        pb2 === pb2 * pb2
        pb3 === pb3 * pb3
        qb0 === qb0 * qb0
        qb1 === qb1 * qb1
        qb2 === qb2 * qb2
        qb3 === qb3 * qb3
        pb01 <== pb0 + 2 * pb1
        pb012 <== pb01 + 4 * pb2
        p <== pb012 + 8 * pb3
        qb01 <== qb0 + 2 * qb1
        qb012 <== qb01 + 4 * qb2
        q <== qb012 + 8 * qb3
        n <== p * q
    """
    public = [91]
    vk = c.make_verification_key(setup, 16, eqs)
    print("Generated verification key")
    assignments = c.fill_variable_assignments(eqs, {
        'pb3': 1, 'pb2': 1, 'pb1': 0, 'pb0': 1,
        'qb3': 0, 'qb2': 1, 'qb1': 1, 'qb0': 1,
    })
    proof = p.prove_from_witness(setup, 16, eqs, assignments)
    print("Generated proof")
    assert v.verify_proof(setup, 16, vk, proof, public, optimized=True)
    print("Factorization test success!")

if __name__ == '__main__':
    setup = basic_test()
    ab_plus_a_test(setup)
    one_public_input_test(setup)
    proof = prover_test(setup)
    verifier_test(setup, proof)
    factorization_test(setup)
