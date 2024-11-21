#### Disclaimer

DO NOT USE FOR ANYTHING IN REAL LIFE. DO NOT ASSUME THE PROTOCOL DESCRIBED HERE IS SOUND. TALK TO A SPECIALIST IF YOU'RE LOOKING TO USE STARKS IN YOUR APPLICATION.

#### What is this?

See main article: https://vitalik.ca/general/2018/07/21/starks_part_3.html

This is a very basic implementation of a STARK on a MIMC computation that is probably (ie. definitely) broken in a few places but is intended as a proof of concept to show the rough level of complexity that is involved in implementing a simple STARK. A STARK is a really cool proof-of-computation scheme that allows you to create an efficiently verifiable proof that some computation was executed correctly; the verification time only rises logarithmically with the computation time, and that relies only on hashes and information theory for security.

The STARKs are done over a finite field chosen to have `2**32`'th roots of unity (to facilitate STARKs), and NOT have 3rd roots of unity (to facilitate MIMC). The MIMC permutation in general takes the form:

                           k_1                     k_2
                            |                       |
                            v                       v
    x0 ---> (x->x**3) ---> xor ---> (x->x**3) ---> xor --- ... ---> output

Where the `k_i` values are round constants. MIMC can be used as a building block in a hash function, or as a verifiable delay function; its simple arithmetic representation makes it ideal for use in STARKs, SNARKs, MPC and other "cryptography over general-purpose computation" schemes.

The MIMC round constants used here are successive powers of 9 mod `2**256` xored with 1, though could be set to anything. The computational trace is computed over successive powers of a `2**k`'th root of unity. This allows the constraint checking polynomial to be `C(P(x), P(r*x), K(x)) = P(r*x) - P(x)**3 - K(x)`.

For a description of how STARKs work, see:

* [STARKs, part 1: Proofs with Polynomials](https://vitalik.ca/general/2017/11/09/starks_part_1.html)
* [STARKs, part 2: Thank Goodness it's FRI-day](https://vitalik.ca/general/2017/11/22/starks_part_2.html)

For more discussion on MIMC, see:

* [Zcash issue 2233](https://github.com/zcash/zcash/issues/2233)

### How does the STARK scheme here work?

Here are the approximate steps in the code. Note that all arithmetic is done over the finite field of integers mod `2**256 - 2**32 * 351 + 1`.

1. Let `P[0]` equal the input, and `P[i+1] = P[i]**3 + K[i]`, up to `steps` (where `steps` must be a power of 2)
2. Construct a polynomial where `P(subroot ^ i) = P[i]` up to steps-1, where `subroot` is a steps'th root of unity (that is, `subroot**steps = 1`). Do the same with K.
3. Construct the polynomial `CP(x) = C(P(x), P(x * subroot), K(x))`. Note that since `P(x * subroot) = P(x) ^ 3 + K(x), CP(x) = 0` for any x inside the computation trace (that is, powers of subroot except the last).
4. Construct the polynomial `Z(x)`, which is the minimal polynomial that is 0 across the computation trace. Note that because Z is minimal, CP must be a multiple of Z.
5. Construct `D = CP / Z`. Because CP is a multiple of Z, D must itself be a low-degree polynomial.
6. Put D and P into Merkle trees.
7. Construct `L = D + k * x**steps * P`, where `k` is selected based on the Merkle roots of D and P.
8. Create an FRI proof that L is low-degree ((7) and (8) together are a clever way of making an aggregate low-degree proof of D and P)
9. Create probabilistic checks, using the Merkle root of L as source data, to spot-check that D(x) * Z(x) actually equal C(P(x)).

The probabilistic checks and the FRI proof are themselves restricted to `2**precision`'th roots of unity, where `precision = 8 * steps` (so we're FRI checking that the degree of L, which equals twice the degree of P, is at most 1/4 the theoretical maximum).
