#ifndef LIMBO_PARALLEL_HPP_
#define LIMBO_PARALLEL_HPP_

#include <vector>

#ifdef USE_TBB
#include <tbb/concurrent_vector.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#endif

namespace par {
#ifdef USE_TBB
  template <typename X>
  using vector = tbb::concurrent_vector<X>;

  template<typename V>
  std::vector<typename V::value_type> convert_vector(const V& v) {
    std::vector<typename V::value_type> v2(v.size());
    std::copy(v.begin(), v.end(), v2.begin());
    return v2;
  }
#else
  template <typename X>
  using vector = std::vector<X>;

  template<typename V>
  V convert_vector(const V& v) {
    return v;
  }

#endif


#ifdef USE_TBB
  inline void init() {
    static tbb::task_scheduler_init init;
  }
#else
  void init() {
  }
#endif

  // parallel for
  template<typename F>
  inline void loop(size_t begin, size_t end, const F& f) {
#ifdef USE_TBB
    tbb::parallel_for(size_t(begin), end, size_t(1), [&](size_t i) {
      f(i);
    });
#else
    for (size_t i = begin; i < end; ++i) f(i);
#endif
  }

  template<typename T1, typename T2, typename T3>
  inline void sort(T1 i1, T2 i2, T3 comp) {
#ifdef USE_TBB
    tbb::parallel_sort(i1, i2, comp);
#else
    std::sort(i1, i2, comp);
#endif
  }

  // replicate a function nb times
  template<typename F>
  inline void replicate(size_t nb, const F& f) {
#ifdef USE_TBB
    tbb::parallel_for(size_t(0), nb, size_t(1), [&](size_t i) {
      f();
    });
#else
    for (size_t i = 0; i < nb; ++i) f();
#endif
  }

}

#endif
