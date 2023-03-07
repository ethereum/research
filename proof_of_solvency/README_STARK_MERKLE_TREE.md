This demo is based on Vitalik's proof of solvency proposal and implemented using the STARK proof system. 

See https://vitalik.ca/general/2022/11/19/proof_of_solvency.html

It provides users with proofs that constrain the sum of all assets and the non-negativity of their net asset value. 

This is a basic version of the solution.

Most of the modules used here including fft and fri etc. came from "../mimc_stark".


It contains three core constraint formulas:

1. Total amount constraint
$$I(z^{uts-1})=I(z^{uts-2})-sum/N$$

$$I(z^{uts*(i+1)+(uts-1)}) = I(z^{uts*(i+1) +(uts-2)}) + I(z^{uts*i+(uts-1)}) - sum/N, i < N-1$$

$$I(z^{uts*(N-1)+(uts-1)})=0$$

uts: user trace size, the number of traces included in each user's data

sum: total user balance

N: number of users

2. Non-negativity constraint：
$$(I(z^{i+1}) - 2 * I(z^{i})) * (I(z^{i+1}) - 2 * I(z^{i}) -1) = 0, i mod uts \notin \{ uts-1, uts-2 \}$$
$$I(z^{i*uts})=0, i < N$$

3. Include a positivity constraint：
$$I(z^{uts * i+(uts-2)})=balance(z^{uts * i+(uts-2)}),i<N$$