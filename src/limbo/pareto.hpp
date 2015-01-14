#ifndef PARETO_HPP
#define PARETO_HPP

#include <algorithm>
#include "limbo/parallel.hpp"

namespace pareto {
  namespace impl {

    // returns :
    //  1 if i1 dominates i2
    // -1 if i2 dominates i1
    // 0 if both a and b are non-dominated
    template<typename T>
    static int dominate_flag(const T& i1, const T& i2) {

      size_t nb_objs = i1.size();
      assert(nb_objs);

      bool flag1 = false, flag2 = false;
      for (unsigned i = 0; i < nb_objs; ++i) {
        if (i1(i) > i2(i))
          flag1 = true;
        else if (i2(i) > i1(i))
          flag2 = true;
      }
      if (flag1 && !flag2)
        return 1;
      else if (!flag1 && flag2)
        return -1;
      else
        return 0;
    }

    // true if i1 dominate i2
    template<typename T>
    inline bool dominate(const T & i1, const T & i2)  {
      return (dominate_flag(i1, i2) == 1);
    }

    template<int K, typename T, typename T2>
    static bool non_dominated(const T& p_objs, const T2& objs)  {
      for (auto x : objs)
        if (dominate(std::get<K>(x), p_objs))
          return false;
      return true;
    }


    // lexical order
    template<int K>
    struct compare_objs_lex {
      compare_objs_lex() {}
      template<typename T>
      bool operator()(const T& i1, const T& i2) const {
        for (int i = 0; i < std::get<K>(i1).size(); ++i)
          if (std::get<K>(i1)(i) > std::get<K>(i2)(i))
            return true;
          else if (std::get<K>(i1)(i) < std::get<K>(i2)(i))
            return false;
        return false;
      }
    };

    template<typename T>
    inline std::vector<T> new_vector(const T& t) {
      std::vector<T> v;
      v.push_back(t);
      return v;
    }
    template<int K>
    struct comp_fronts {
      // this functor is ONLY for sort2objs
      template<typename T>
      bool operator()(const T& f2, const T& f1) const {
        assert(f1.size() == 1);
        assert(std::get<K>(f1[0]).size() == 2);
        // we only need to compare f1 to the value of the last element of f2
        if (std::get<K>(f1[0])(1) < std::get<K>(f2.back())(1))
          return true;
        else
          return false;
      }
    };

    // O(n^2) procedure, for > 2 objectives
    template<int K, typename T>
    T pareto_set_std(const T& p) {
#ifdef __GXX_EXPERIMENTAL_CXX0X__
      typename par::vector<typename T::value_type>::type pareto; // old fashion way to create template alias (for GCC 4.6...)
#else
      par::vector<typename T::value_type> pareto; // Using Template alias (for GCC 4.7 and later)
#endif
      par::loop(0, p.size(), [&](size_t i) {
        if (i % 10000 == 0) {
          std::cout << i << '[' << p.size() << "] ";
          std::cout.flush();
        }
        if (non_dominated<K>(std::get<K>(p[i]), p))
          pareto.push_back(p[i]);
      });
      std::sort(pareto.begin(), pareto.end(), compare_objs_lex<K>());
      return par::convert_vector(pareto);
    }

    // O(n lg n), for 2 objectives ONLY
    // see M. T. Jensen, 2003
    template<int K, typename T>
    T sort_2objs(const T& v) {
      T p = v;
      par::sort(p.begin(), p.end(), compare_objs_lex<K>());

      std::vector<T> f;
      f.push_back(impl::new_vector(p[0]));
      size_t e = 0;
      for (size_t i = 1; i < p.size(); ++i) {
        if (i % 10000 == 0) {
          std::cout << i << " [" << p.size() << "] ";
          std::cout.flush();
        }
        if (std::get<K>(p[i])(1) > std::get<K>(f[e].back())(1)) { // !dominate(si, f_e)
          auto b = std::lower_bound(f.begin(), f.end(), impl::new_vector(p[i]),
                                    impl::comp_fronts<K>());
          assert(b != f.end());
          b->push_back(p[i]);
        } else {
          ++e;
          f.push_back(impl::new_vector(p[i]));
        }
      }
      return f[0];
    }
  }

  // argument vector of P (std::vector<P>)
  // where P is a tuple with the objective values in std::get<K>(p);
  template<int K, typename T>
  static T pareto_set(const T& v) {
    assert(v.size());
    size_t nb_objs = std::get<K>(v[0]).size();
    assert(nb_objs > 1);
    /*  if (nb_objs == 2)
        return impl::sort_2objs<K>(v);
      else*/
    return impl::pareto_set_std<K>(v);
  }
}

#endif
