#ifndef LIMBO_TOOLS_MACROS_HPP
#define LIMBO_TOOLS_MACROS_HPP

#include <boost/algorithm/string.hpp>
#include <Eigen/Core>

#define BO_PARAM(Type, Name, Value) \
    static constexpr Type Name() { return Value; }

#define BO_REQUIRED_PARAM(Type, Name)                                         \
    static const Type Name()                                                  \
    {                                                                         \
        static_assert(false, "You need to define the parameter:" #Name " !"); \
        return Type();                                                        \
    }

#define BO_DYN_PARAM(Type, Name)           \
    static Type _##Name;                   \
    static Type Name() { return _##Name; } \
    static void set_##Name(const Type& v) { _##Name = v; }

#define BO_DECLARE_DYN_PARAM(Type, Namespace, Name) Type Namespace::_##Name;

#define __VA_NARG__(...) (__VA_NARG_(_0, ##__VA_ARGS__, __RSEQ_N()) - 1)
#define __VA_NARG_(...) __VA_ARG_N(__VA_ARGS__)
#define __VA_ARG_N(                                   \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,          \
    _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, \
    _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, \
    _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, \
    _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, \
    _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, \
    _61, _62, _63, N, ...) N
#define __RSEQ_N()                              \
    63, 62, 61, 60,                             \
        59, 58, 57, 56, 55, 54, 53, 52, 51, 50, \
        49, 48, 47, 46, 45, 44, 43, 42, 41, 40, \
        39, 38, 37, 36, 35, 34, 33, 32, 31, 30, \
        29, 28, 27, 26, 25, 24, 23, 22, 21, 20, \
        19, 18, 17, 16, 15, 14, 13, 12, 11, 10, \
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define BO_PARAM_ARRAY(Type, Name, ...)                  \
    static Type Name(size_t i)                           \
    {                                                    \
        assert(i < __VA_NARG__(__VA_ARGS__));            \
        static constexpr Type _##Name[] = {__VA_ARGS__}; \
        return _##Name[i];                               \
    }                                                    \
    static constexpr size_t Name##_size()                \
    {                                                    \
        return __VA_NARG__(__VA_ARGS__);                 \
    }                                                    \
    typedef Type Name##_t;

#define BO_PARAM_VECTOR(Type, Name, ...)                                                    \
    static const Eigen::Matrix<Type, __VA_NARG__(__VA_ARGS__), 1> Name()                    \
    {                                                                                       \
        static constexpr Type _##Name[] = {__VA_ARGS__};                                    \
        return Eigen::Map<const Eigen::Matrix<Type, __VA_NARG__(__VA_ARGS__), 1>>(_##Name); \
    }

#define BO_PARAM_STRING(Name, Value) \
    static constexpr const char* Name() { return Value; }

#define BO_PARAMS(Stream, P)                                  \
    struct Ps__ {                                             \
        Ps__()                                                \
        {                                                     \
            static std::string __params = #P;                 \
            boost::replace_all(__params, ";", ";\n");         \
            boost::replace_all(__params, "{", "{\n");         \
            Stream << "Parameters:" << __params << std::endl; \
        }                                                     \
    };                                                        \
    P;                                                        \
    static Ps__ ____p;

#endif
