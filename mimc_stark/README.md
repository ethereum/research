#### What is this?

This is a very basic implementation of a STARK on a MIMC computation that is probably (ie. definitely) broken in a few places but is intended as a proof of concept to show the rough level of complexity that is involved in implementing a simple STARK. A STARK is a really cool proof-of-computation scheme that allows you to create an efficiently verifiable proof that some computation was executed correctly; the verification time only rises logarithmically with the computation time, and that relies only on hashes and information theory for security.

The STARKs are done over a finite field chosen to have 2^32'th roots of unity (to facilitate STARKs), and NOT have 3rd roots of unity (to facilitate MIMC). The MIMC permutation in general takes the form:

                          k_1                    k_2
                           |                      |
                           v                      v
    x0 ---> (x->x^3) ---> xor ---> (x->x^3) ---> xor --- ... ---> output

Where the `k_i` values are round constants. MIMC can be used as a building block in a hash function, or as a verifiable delay function; its simple arithmetic representation makes it ideal for use in STARKs, SNARKs, MPC and other "cryptography over general-purpose computation" schemes.

The MIMC round constants used here are 2^k'th roots of unity for k <= 32. The computational trace is computed over successive powers of a 2^k'th root of unity. This allows the constraint checking polynomial to be `C(P(x), P(r*x), x) = P(r*x) - P(x)**3 - x`.

For a description of how STARKs work, see:

* [STARKs, part 1: Proofs with Polynomials](https://vitalik.ca/general/2017/11/09/starks_part_1.html)
* [STARKs, part 2: Thank Goodness it's FRI-day](https://vitalik.ca/general/2017/11/22/starks_part_2.html)

For more discussion on MIMC, see:

* [Zcash issue 2233](https://github.com/zcash/zcash/issues/2233)

The implementation is definitely suboptimal. For example, one optimization would be to reduce the two FRIs to one by low-degree proving a random linear combination k1 * D + k2 * P^2; this could cut the verification time by close to half. Another would be to remove redundancies in computing zero polynomials in multiple places; a more sophisticated optimization would be to move to subquadratic algorithms (Karatsuba or ideally FFT-based) for polynomial interpolation. The purpose is for the code to be deliberately simple to aid with understanding.
