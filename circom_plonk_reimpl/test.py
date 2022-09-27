import circom_tools as c
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
    print("Success")
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
    print("Success")

if __name__ == '__main__':
    setup = basic_test()
    ab_plus_a_test(setup)
