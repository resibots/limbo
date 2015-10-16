#ifndef _BO_MACROS_HPP_
#define _BO_MACROS_HPP_

#define BO_PARAM(Type, Name, Value) \
    static constexpr Type Name() { return Value; }

#define BO_REQUIRED_PARAM(Type, Name)                                          \
    static constexpr Type Name()                                               \
    {                                                                          \
        static_assert(false, "You need to define the parameter:"##Name##" !"); \
        return Type();                                                         \
    }

#define BO_DYN_PARAM(Type, Name)           \
    static Type _##Name;                   \
    static Type Name() { return _##Name; } \
    static void set_##Name(const Type& v) { _##Name = v; }

#define BO_DECLARE_DYN_PARAM(Type, Namespace, Name) Type Namespace::_##Name;

#define BO_PARAM_ARRAY(Type, Name, ...)                  \
    static Type Name(size_t i)                           \
    {                                                    \
        assert(i < Name##_size());                       \
        static constexpr Type _##Name[] = {__VA_ARGS__}; \
        return _##Name[i];                               \
    }                                                    \
    static size_t Name##_size()                          \
    {                                                    \
        static constexpr Type _##Name[] = {__VA_ARGS__}; \
        return sizeof(_##Name) / sizeof(Type);           \
    }                                                    \
    typedef Type Name##_t;

#define BO_STRING(N, V) static constexpr const char* N() { return V; }

#endif
