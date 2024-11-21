#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Requirements:
- I/O bound: cycles spent on I/O â‰« cycles spent in cpu
- no sharding: impossible to implement data locality strategy
- easy verification

Thoughts:

Efficient implementations will not switch context (threading) when waiting for data.
But they would leverage all fill buffers and have concurrent memory accesses.
It can be assumed, that code can be written in a way to calculate N (<10)
nonces in parallel (on a single core).

So, after all maybe memory bandwidth rather than latency is the actual bottleneck.
Can this be solved in a way that aligns with hashing nonces and allows
for a quick verification? Probably not.

Loop unrolling:
Initially proposed dagger sets offer data locality which allows to scale the algo 
on multiple cores/l2chaches. 320MB / 40sets = 8MB (< L2 cache)
A solution is to make accessed mem location depended on the value of the
previous access.

Partitial Memory:
If a users only keeps e.g. one third of each DAG in memory (i.e. to 
have in L3 cache), he still can answer ~0.5**k of accesses by substituting 
them through previous node lookups. 
This can be mitigated by
a) making each node deterministically depend on the value of at
least one close high memory node. Optionally for quick validation, select
the 2nd dependency for the lower (cached) memory. see produce_dag_k2dr
b) for DAG creation, using a hashing function which needs more cycles
than multiple memory lookups would - even for GPUs/FPGAs/ASICs.
"""


import time

from pyethereum import utils


def decode_int(s):
    o = 0
    for i in range(len(s)):
        o = o * 256 + ord(s[i])
    return o


def encode_int(x):
    o = ''
    for _ in range(64):
        o = chr(x % 256) + o
        x //= 256
    return o


def sha3(x):
    return decode_int(utils.sha3(x))


def cantor_pair(x, y, p):
    return ((x+y) * (x+y+1) / 2 + y) % p


def get_daggerset(params, seedset):
    return [produce_dag(params, i) for i in seedset]


def update_daggerset(params, daggerset, seedset, seed):
    idx = decode_int(seed) % len(daggerset)
    seedset[idx] = seed
    daggerset[idx] = produce_dag(params, seed)


def produce_dag(params, seed):
    k, hk, w, hw, n, p, t = params.k, params.hk, params.w, \
        params.hw, params.dag_size, params.p, params.h_threshold
    print 'Producing dag of size %d (%d memory)' % (n, n * params.wordsz)
    o = [sha3(seed)]
    init = o[0]
    picker = 1
    for i in range(1, n):
        x = 0
        picker = (picker * init) % p
        curpicker = picker
        if i < t:
            for j in range(k):  # can be flattend if params are known
                x ^= o[curpicker % i]
                curpicker >>= 10
        else:
            for j in range(hk):
                x ^= o[curpicker % t]
                curpicker >>= 10
        o.append(pow(x, w if i < t else hw, p))  # use any "hash function" here
    return o


def quick_calc(params, seed, pos, known=None):
    k, hk, w, hw, p, t = params.k, params.hk, params.w, \
        params.hw, params.p, params.h_threshold
    init = sha3(seed) % p
    if known is None:
        known = {}
    known[0] = init

    def calc(i):
        if i not in known:
            curpicker = pow(init, i, p)
            x = 0
            if i < t:
                for j in range(k):
                    x ^= calc(curpicker % i)
                    curpicker >>= 10
                known[i] = pow(x, w, p)
            else:
                for j in range(hk):
                    x ^= calc(curpicker % t)
                    curpicker >>= 10
                known[i] = pow(x, hw, p)
        return known[i]
    o = calc(pos)
    print 'Calculated index %d in %d lookups' % (pos, len(known))
    return o


def hashimoto(params, daggerset, header, nonce):
    """
    Requirements:
    - I/O bound: cycles spent on I/O â‰« cycles spent in cpu
    - no sharding: impossible to implement data locality strategy

    # I/O bound:
    e.g. lookups = 16
    sha3:       12 * 32   ~384 cycles
    lookups:    16 * 160 ~2560 cycles # if zero cache
    loop:       16 * 3     ~48 cycles
    I/O / cpu = 2560/432 = ~ 6/1

    # no sharding
    lookups depend on previous lookup results
    impossible to route computation/lookups based on the initial sha3
    """
    rand = sha3(header + encode_int(nonce)) % params.p
    mix = rand
    # loop, that can not be unrolled
    # dag and dag[pos] depended on previous lookup
    for i in range(params.lookups):
        v = mix if params.is_serial else rand >> i
        dag = daggerset[v % params.num_dags]  # modulo
        pos = v % params.dag_size    # modulo
        mix ^= dag[pos]         # xor
        # print v % params.num_dags, pos, dag[pos]
    print header, nonce, mix
    return mix


def light_hashimoto(params, seedset, header, nonce):
    rand = sha3(header + encode_int(nonce)) % params.p
    mix = rand

    for i in range(params.lookups):
        v = mix if params.is_serial else rand >> i
        seed = seedset[v % len(seedset)]
        pos = v % params.dag_size
        qc = quick_calc(params, seed, pos)
        # print v % params.num_dags, pos, qc
        mix ^= qc
    print 'Calculated %d lookups' % \
        (params.lookups)
    print header, nonce, mix
    return mix


def light_verify(params, seedset, header, nonce):
    h = light_hashimoto(params, seedset, header, nonce)
    return h <= 256**params.wordsz / params.diff


def mine(daggerset, params, header, nonce=0):
    orignonce = nonce
    origtime = time.time()
    while 1:
        h = hashimoto(params, daggerset, header, nonce)
        if h <= 256**params.wordsz / params.diff:
            noncediff = nonce - orignonce
            timediff = time.time() - origtime
            print 'Found nonce: %d, tested %d nonces in %.2f seconds (%d per sec)' % \
                (nonce, noncediff, timediff, noncediff / timediff)
            return nonce
        nonce += 1


class params(object):
    """
    === tuning ===
    memory: memory requirements â‰« L2/L3/L4 cache sizes
    lookups:  hashes_per_sec(lookups=0) â‰« hashes_per_sec(lookups_mem_hard)
    k:        ?
    d:        higher values enfore memory availability but require more quick_calcs
    num_dags: so that a dag can be updated in reasonable time
    """
    p = (2 ** 256 - 4294968273)**2    # prime modulus
    wordsz = 64                       # word size
    memory = 10 * 1024**2            # memory usage
    num_dags = 2                     # number of dags
    dag_size = memory/num_dags/wordsz # num 64byte values per dag
    lookups = 40                      # memory lookups per hash
    diff = 2**14                      # higher is harder
    k = 2                             # num dependecies of each dag value
    hk = 8                            # dependencies for final nodes
    d = 8                             # max distance of first dependency (1/d=fraction of size)
    w = 2                             # work factor on node generation
    hw = 8                            # work factor on final node generation
    h_threshold = dag_size*2/5        # cutoff between final and nonfinal nodes
    is_serial = False                 # hashimoto is serial


if __name__ == '__main__':
    print dict((k, v) for k, v in params.__dict__.items()
               if isinstance(v, int))

    # odds of a partitial storage attack
    missing_mem = 0.01
    P_partitial_mem_success = (1-missing_mem) ** params.lookups
    print 'P success per hash with %d%% mem missing: %d%%' % \
        (missing_mem*100, P_partitial_mem_success*100)

    # which actually only results in a slower mining,
    # as more hashes must be tried
    slowdown = 1 / P_partitial_mem_success
    print 'x%.1f speedup required to offset %d%% missing mem' % \
        (slowdown, missing_mem*100)

    # create set of DAGs
    st = time.time()
    seedset = [str(i) for i in range(params.num_dags)]
    daggerset = get_daggerset(params, seedset)
    print 'daggerset with %d dags' % len(daggerset), 'size:', \
        64*params.dag_size*params.num_dags / 1024**2, 'MB'
    print 'creation took %.2fs' % (time.time() - st)

    # update DAG
    st = time.time()
    update_daggerset(params, daggerset, seedset, seed='qwe')
    print 'updating 1 dag took %.2fs' % (time.time() - st)

    # Mine
    for i in range(1):
        header = 'test%d' % i
        print '\nmining', header
        nonce = mine(daggerset, params, header)
        # verify
        st = time.time()
        assert light_verify(params, seedset, header, nonce)
        print 'verification took %.2fs' % (time.time() - st)
