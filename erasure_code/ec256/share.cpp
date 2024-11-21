#include "share.h"

// returns a * (x+1) in the Galois field
// (since (x+1) is a primitive root)
static constexpr std::uint8_t tpl(std::uint8_t a) {
    return a ^ (a<<1) // a * (x+1)
        ^ ((a & (1<<7)) != 0
           ? // would overflow (have an x^8 term); reduce by the AES polynomial,
             // x^8 + x^4 + x^3 + x + 1
             0b00011011u
           : 0
          );
}

// constexpr functions to compute exp/log
// these are not intended to be fast, but they must be constexpr to populate a
// table at compile time
static constexpr std::uint8_t gexp(unsigned k) {
    return k > 0 ? tpl(gexp(k-1)) : 1;
}
static constexpr std::uint8_t glog(unsigned k, unsigned i = 0, unsigned v = 1) {
    return k == v ? i : glog(k, i+1, tpl(v));
}

// insane hack (courtesy of Xeo on stackoverflow): gen_seq<N> expands to a
// struct that derives from seq<0, 1, ..., N-1>
template<unsigned... I> struct seq{};
template<unsigned N, unsigned... I>
struct gen_seq : gen_seq<N-1, N-1, I...>{};
template<unsigned... I>
struct gen_seq<0, I...> : seq<I...>{};

// produce the actual tables in array form...
template<unsigned... I>
constexpr std::array<Galois, 255> exptbl(seq<I...>) {
    return { { Galois(gexp(I))... } };
}
template<unsigned... I>
constexpr std::array<std::uint8_t, 256> logtbl(seq<I...>) {
    // manually populate entry zero, for two reasons:
    // - it makes glog simpler
    // - it avoids clang++'s default template instantiation depth limit of 256
    return { { 0, glog(I+1)... } };
}

// and initialize the static variables
const std::array<Galois, 255> Galois::exptable = exptbl(gen_seq<255>{});
const std::array<std::uint8_t, 256> Galois::logtable = logtbl(gen_seq<255>{});

// by populating everything at compile-time, we avoid a static initialization
// step and any possible associated static initialization "races"
