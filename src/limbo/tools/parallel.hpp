#ifndef LIMBO_TOOLS_PARALLEL_HPP
#define LIMBO_TOOLS_PARALLEL_HPP

#include <vector>
#include <algorithm>

#ifdef USE_TBB
#include <tbb/concurrent_vector.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#endif

namespace limbo {
    namespace tools {
        namespace par {
#ifdef USE_TBB
#ifdef __GXX_EXPERIMENTAL_CXX0X__
            template <typename X> // old fashion way to create template alias (for GCC
            // 4.6...)
            struct vector {
                typedef tbb::concurrent_vector<X> type;
            };
#else
            template <typename X>
            using vector = tbb::concurrent_vector<X>; // Template alias (for GCC 4.7 and later)
#endif
            template <typename V>
            std::vector<typename V::value_type> convert_vector(const V& v)
            {
                std::vector<typename V::value_type> v2(v.size());
                std::copy(v.begin(), v.end(), v2.begin());
                return v2;
            }
#else
#ifdef __GXX_EXPERIMENTAL_CXX0X__
            template <typename X> // old fashion way to create template alias (for GCC
            // 4.6...)
            struct vector {
                typedef std::vector<X> type;
            };
#else
            template <typename X>
            using vector = std::vector<X>; // Template alias (for GCC 4.7 and later)
#endif

            template <typename V>
            V& convert_vector(const V& v)
            {
                return v;
            }

#endif

#ifdef USE_TBB
            inline void init()
            {
                static tbb::task_scheduler_init init;
            }
#else
            void init()
            {
            }
#endif

            // parallel for
            template <typename F>
            inline void loop(size_t begin, size_t end, const F& f)
            {
#ifdef USE_TBB
                tbb::parallel_for(size_t(begin), end, size_t(1), [&](size_t i) {
    // clang-format off
                f(i);
                    // clang-format on
                });
#else
                for (size_t i = begin; i < end; ++i)
                    f(i);
#endif
            }

            template <typename T, typename F, typename C>
            T max(const T& init, int num_steps, const F& f, const C& comp)
            {
#ifdef USE_TBB
                auto body = [&](const tbb::blocked_range<size_t>& r, T current_max) -> T {
    // clang-format off
            for (size_t i = r.begin(); i != r.end(); ++i)
            {
                T v = f(i);
                if (comp(v, current_max))
                  current_max = v;
            }
            return current_max;
                    // clang-format on
                };
                auto joint = [&](const T& p1, const T& p2) -> T {
    // clang-format off
            if (comp(p1, p2))
                return p1;
            return p2;
                    // clang-format on
                };
                return tbb::parallel_reduce(tbb::blocked_range<size_t>(0, num_steps), init,
                    body, joint);
#else
                T current_max = init;
                for (size_t i = 0; i < num_steps; ++i) {
                    T v = f(i);
                    if (comp(v, current_max))
                        current_max = v;
                }
                return current_max;
#endif
            }

            template <typename T1, typename T2, typename T3>
            inline void sort(T1 i1, T2 i2, T3 comp)
            {
#ifdef USE_TBB
                tbb::parallel_sort(i1, i2, comp);
#else
                std::sort(i1, i2, comp);
#endif
            }

            // replicate a function nb times
            template <typename F>
            inline void replicate(size_t nb, const F& f)
            {
#ifdef USE_TBB
                tbb::parallel_for(size_t(0), nb, size_t(1), [&](size_t i) {
    // clang-format off
                f();
                    // clang-format on
                });
#else
                for (size_t i = 0; i < nb; ++i)
                    f();
#endif
            }
        }
    }
}

#endif
