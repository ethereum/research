"""
CSIDH [1, 2] adaptation in Sage of the Whisk [4] commitment scheme discussed in Part 1 of [5].
References:
[1] https://github.com/KULeuven-COSIC/CSI-FiSh/blob/master/implementation/supersingular.sage
[2] https://eprint.iacr.org/2019/498.pdf
[3] https://github.com/ethereum/consensus-specs/pull/2800/files
[4] https://ethresear.ch/t/whisk-a-practical-shuffle-based-ssle-protocol-for-ethereum/11763
[5] https://crypto.ethereum.org/blog/pq-ssle
"""
import numpy as np 
import hashlib 
import random 
from decimal import *
from decimal import ROUND_HALF_UP
import relation_lattices

"""
Methods in CSIDH class obtained from https://github.com/KULeuven-COSIC/CSI-FiSh/blob/master/implementation/supersingular.sage
"""
class CSIDH: 
    def __init__(self, ls):
        self.ls = ls
        self.p=4*prod(ls)-1
        self.max_exp = ceil((sqrt(self.p) ** (1/len(ls)) - 1) / 2)
        self.base = GF(self.p)(0)
        Fp2.<i> = GF(self.p**2,modulus=x**2+1)
        self.Fp2 = Fp2
        
    def montgomery_curve(self, A):
        return EllipticCurve(self.Fp2, [0, A, 0, 1, 0])
        
    def montgomery_coefficient(self,E):
        Ew = E.change_ring(GF(self.p)).short_weierstrass_model()
        _, _, _, a, b = Ew.a_invariants()
        R.<z> = GF(self.p)[]
        r = (z**3 + a*z + b).roots(multiplicities=False)[0]
        s = sqrt(3 * r**2 + a)
        if not is_square(s): s = -s
        A = 3 * r / s
        assert self.montgomery_curve(A).change_ring(GF(self.p)).is_isomorphic(Ew)
        return GF(self.p)(A)

    def private(self):
        return [randrange(-self.max_exp, self.max_exp + 1) for _ in range(len(ls))]

    def validate(self, A):
        while True:
            k = 1
            P = self.montgomery_curve(A).lift_x(GF(self.p).random_element())
            for l in self.ls:
                Q = (p + 1) // l * P
                if not Q: continue
                if l * Q: return False
                k *= l
                if k > 4 * sqrt(self.p): return True
            
    def action(self, pub, priv):

        E = self.montgomery_curve(pub)
        es = priv[:]
        while any(es):
            E._order = (self.p + 1)**2 # else sage computes this
            P = E.lift_x(GF(self.p).random_element())
            s = +1 if P.xy()[1] in GF(self.p) else -1
            k = prod(l for l, e in zip(self.ls, es) if sign(e) == s)
            P *= (self.p + 1) // k

            for i, (l, e) in enumerate(zip(self.ls, es)):

                if sign(e) != s: continue

                Q = k // l * P
                if not Q: continue
                Q._order = l # else sage computes this
                phi = E.isogeny(Q)

                E, P = phi.codomain(), phi(P)
                es[i] -= s
                k //= l

        return self.montgomery_coefficient(E)

class CSI_FISH: 
    
    def __init__(self, ls, num_challs, A, B): 

        self.ls = ls 
        self.csidh = CSIDH(self.ls)
        self.t = num_challs # number of challenges used in multi-round ID scheme 
        self.keygen()
        self.class_number = 254652442229484275177030186010639202161620514305486423592570860975597611726191
        self.num_primes = len(ls)
        self.lam=3
        
        #relation lattices 
        self.A = [] 
        self.B = []
        for i in range(0,len(A),self.num_primes):
            a = relation_lattices.A[i:i+self.num_primes]
            self.A.append([Decimal(str(val)) for val in a])
        for i in range(0,len(B),self.num_primes):
            b = relation_lattices.B[i:i+self.num_primes]
            self.B.append([Decimal(str(val)) for val in b])
      
    """
    Initialize public keys and secret key.
    E_0 and [a]*E_0 are the public keys and a is the secret key.  
    """
              
    def keygen(self): 
        self.E_0 = self.csidh.base
        self.__es = self.csidh.private() # private variable 
        self.E_A = self.csidh.action(self.E_0, self.__es)
        
    """
    Babai Rounding.  
    """
    def babai_rounding(self,L, target):
        lattice_matrix = []
        for i in range(0,len(L),self.num_primes):
            lattice_matrix.append(L[i:i+self.num_primes])
        lattice_matrix = np.array(lattice_matrix).astype(float)
        lattice_matrix=np.transpose(lattice_matrix)
        lattice_matrix_inv = np.linalg.inv(lattice_matrix)
        closest_unrounded=np.dot(lattice_matrix_inv, np.array(target))
        cvp_solution = np.dot(lattice_matrix, np.round(closest_unrounded))
        return cvp_solution

    """
    Babai nearest plane algorithm. 
    Given a target vector finds the nearest vector wrt  
    L1 norm in the linear subspace defined by the relation lattice self.A
    See [2] Section 4 and https://github.com/KULeuven-COSIC/CSI-FiSh/blob/master/implementation/classgroup.c
    """
    
    def babai_nearest_vector(self, target): 
        target_copy = target.copy()
        for i in range(self.num_primes-1, -1, -1): 
            ip1 = np.dot(target_copy, self.B[i])
            ip1 = (ip1/Decimal(IPStrings[i])).to_integral_value(rounding=ROUND_HALF_UP)
            ip1_vec = [ip1 for i in range(0, 74)]
            target_copy -= np.array(self.A[i])*np.array(ip1_vec)
        return target_copy 
    
    
    """
    One round Identification scheme. 
    The prover recieves a challenge c and computes E = [b - a*c]*E_c
    where a is the secret key and b is randomly sampled mod class number. The prover sends H(E)
    and the reduced value of r = b - a*c . 
    Soundness 1/2. See Figure 2 of [2] 
    """
    def identification(self, challenge):
        h = hashlib.sha256()
        #b = random.randint(0, self.class_number) #uncomment when big int issues fixed 
        b=random.randint(0, 100) #for testing 
        target = self.num_primes*[Decimal('0')]
        target[0]=Decimal(str(b))
        z=self.babai_nearest_vector(target)
        reduced_exponent = [int(z_) for z_ in z]
        challenge_curve=None
        final_exponent=None 
        if challenge==0: 
            challenge_curve = self.csidh.action(self.E_0,reduced_exponent)
            final_exponent=reduced_exponent
        elif challenge==1: 
            reduced_exp_minus_a = [reduced_exponent[i] - self.__es[i] for i in range(len(self.__es))]
            final_exponent=reduced_exp_minus_a
            challenge_curve = self.csidh.action(self.E_A,reduced_exp_minus_a)  
        s = str(challenge_curve)
        h.update(s.encode())
        return (challenge, final_exponent, h.digest())
    
    """
    One round verification scheme. 
    The verifier receives a tuple (c, r, H(E)) from the verifier and verifies that indeed 
    H([r]*E_c) == H(E). 
    See Figure 2 of [2] 
    """
    def verify_identification(self, challenge_sent, prover_output):
        h = hashlib.sha256()
        challenge, exponent, hash_s = prover_output
        assert challenge==challenge_sent 
        curve=None
        if challenge==0: 
            curve=self.csidh.action(self.E_0, exponent)
        elif challenge==1: 
            curve=self.csidh.action(self.E_A, exponent)  
        s = str(curve)
        h.update(s.encode())
        if hash_s == h.digest(): 
            return True 
        return False 
    
    """
    DLEQ Multiple round identification scheme. Section 2.4 of https://eprint.iacr.org/2020/1323.pdf. 
    Input: 
    -challenges is a (self.lambda,1) shape array of bits 
    -curves is a (m,2) shape array with tuples (E_i, E_i')
    -sec is an integer mod class number 
    
    Output: 
    -returns a proof pi = (challenges, rs)
    where rs is a (self.lambda, 1) shape array where r_j = b_j - c_j * s
    """
    def dleq_prover(self, challenges, curves, sec):    
        h = hashlib.sha256()
        bs = [] 
        image_curves = [] 
        sec_arr = self.num_primes*[Decimal('0')]
        sec_arr[0]=Decimal(str(sec))
        reduced_sec = [int(z_) for z_ in self.babai_nearest_vector(sec_arr)]
        
        for j in range(self.lam): 
            #b = random.randint(0, self.class_number) #uncomment when big int issues fixed 
            b=random.randint(0, 100)
            target = self.num_primes*[Decimal('0')]
            target[0]=Decimal(str(b))
            z=self.babai_nearest_vector(target)
            reduced_exponent = [int(z_) for z_ in z]
            bs.append(reduced_exponent)
            temp=[]
            for i in range(len(curves)): 
                temp.append(self.csidh.action(curves[i][0],reduced_exponent))
            image_curves.append(temp)
            
        s = ''
        for arr in image_curves: 
            for curve in arr: 
                s += str(curve)
        h.update(s.encode())
        rs = [] 
        for j in range(self.lam): 
            r = [bs[j][i] - challenges[j]*reduced_sec[i] for i in range(len(bs[j]))]
            rs.append(r)
            
        return (challenges, rs, h.digest())
    
    """
    DLEQ verification algorithm. Section 2.4 of https://eprint.iacr.org/2020/1323.pdf. 
    Input: 
    -proof of the form (challenges, rs, hash)
    -curves is a (m,2) shape array with tuples (E_i, E_i')
 
    Output: 
    -returns a 1 if proof verifies correctly, 0 otherwise 
    """   
    def dleq_verifier(self, proof, curves): 
        h = hashlib.sha256()
        challenges = proof[0]
        rs = proof[1]
        hash_s = proof[2]
        image_curves = []
        for j in range(self.lam): 
            if challenges[j]==0: 
                image_curves.append([self.csidh.action(curves[i][0],rs[j]) for i in range(len(curves))])
            if challenges[j]==1: 
                image_curves.append([self.csidh.action(curves[i][1],rs[j]) for i in range(len(curves))])
        s = ''
        for arr in image_curves:
            for curve in arr:
                s+=str(curve)
        h.update(s.encode())
        
        return h.digest()==hash_s
    
    
    """
    Multiple round signature scheme. Soundness (1/2)^t. 
    Inputs is msg to be signed and challenges cs. 
    See Figure 3 of [2] 
    
    """
    def sign(self, msg, cs):
        assert len(cs)==self.t
        h = hashlib.sha256()
        #targets = [random.randint(0, self.class_number) for i in range(self.t)] # uncomment when big int issues fixed 
        targets = [random.randint(0, 100) for i in range(self.t)]
        reduced_exponents=[]
        for t in targets: 
            b = [Decimal('0')]*self.num_primes 
            b[0] = Decimal(str(t))
            z = self.babai_nearest_vector(b)
            reduced_exponents.append([int(z_) for z_ in z])
        final_exponents = []
        challenge_curves=[] 
        for i in range(self.t): #loops through challenges 
            if cs[i]==1: 
                challenge_curves.append(self.csidh.action(self.E_0,reduced_exponents[i]))
                final_exponents.append(reduced_exponents[i])
            elif cs[i]==0: 
                reduced_exp_minus_a = [reduced_exponents[i][j] - self.__es[j] for j in range(len(self.__es))]
                challenge_curves.append(self.csidh.action(self.E_A, reduced_exp_minus_a)) 
                final_exponents.append(reduced_exp_minus_a)
        
        s = ''
        for curve in challenge_curves: # computes H( [r1]*E1 | ... | [r_t]*E_t)
            s += str(curve)
        s+= msg
        h.update(s.encode())
        sig = (final_exponents, cs, h.digest())
        return sig 
    
    """
    Verify multiple round signature scheme. See Figure 3 of [2] 
    """
    
    def verify_signature(self, msg, sig):
        h = hashlib.sha256()
        final_exponents, cs, hash_s = sig
        Es = [] 
        for i in range(self.t): 
            if cs[i]==1: 
                Es.append(self.csidh.action(self.E_0, final_exponents[i]))
            elif cs[i]==0: 
                Es.append(self.csidh.action(self.E_A, final_exponents[i]))
        s = ''
        for curve in Es: 
            s+=str(curve)
        s+= msg
        h.update(s.encode())
        if hash_s == h.digest(): 
            return True 
        return False 
        

##### Example Signature and Verification #########
ls = list(primes(3, 374)) + [587]
p=4*prod(ls)-1
R.<x> = PolynomialRing(GF(p))
fish = CSI_FISH(ls, 3, A,B)
sig=fish.sign("msg", [1,0,1])
fish.verify_signature("msg",sig)

########Test run for DLEQ #######

ls = list(primes(3, 374)) + [587]
p=4*prod(ls)-1
csidh = CSIDH(ls)
fish = CSI_FISH(ls, 3, A,B)
# choose s and three random base curves 
s = random.randint(1, 100)
base_curves = [GF(p)(0) for i in range(2)]
target = fish.num_primes*[Decimal('0')]
target[0]=Decimal(str(s))
z=fish.babai_nearest_vector(target)
reduced_exponent = [int(z_) for z_ in z]
image_curves = [csidh.action(base_curves[i],reduced_exponent) for i in range(2)]
curves = [(base_curves[i], image_curves[i]) for i in range(2)]
print("Proof generated: ")
proof = fish.dleq_prover([1,1,0], curves, s)
print(proof)
print("")
res=fish.dleq_verifier(proof, curves)
print("Result of verification: ")
print(res)
