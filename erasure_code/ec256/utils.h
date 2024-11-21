#ifndef UTILS_H_
#define UTILS_H_

#include <utility>
#include <cstddef>
#include <type_traits>

// turn a pair of iterators into a range
template<typename T>
class iter_pair {
    T a, b;
    typedef typename std::remove_reference<decltype(**static_cast<T*>(nullptr))>::type value_type;
public:
    iter_pair(const T& _a, const T& _b) : a(_a), b(_b) { }
    iter_pair(T&& _a, T&& _b) : a(std::move(_a)), b(std::move(_b)) { }
    T begin() const { return a; }
    T end() const { return b; }

    // for random access iterators
    value_type& operator[](std::size_t ix) { return *(a + ix); }
    const value_type& operator[](std::size_t ix) const { return *(a + ix); }
    std::size_t size() const { return b - a; }
};

template<typename T>
iter_pair<T> make_iter_pair(const T& a, const T& b) {
    return iter_pair<T>(a, b);
}
template<typename T>
iter_pair<T> make_iter_pair(T&& a, T&& b) {
    return iter_pair<T>(std::move(a), std::move(b));
}

#endif
