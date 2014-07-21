#ifndef _BO_MACROS_HPP_
#define _BO_MACROS_HPP_

#define BO_PARAM(Type, Name, Value) static constexpr Type Name() { return Value; }

#define BO_REQUIRED_PARAM(Type, Name) static constexpr Type Name() { static_assert(false, "You need to define the parameter:"##Name##" !"); return Type(); }

#define BO_DYN_PARAM(Type, Name)                           \
    static Type _##Name;                                   \
    static Type Name() { return _##Name; }                 \
    static void set_##Name(const Type& v) { _##Name = v; }

#define BO_DECLARE_DYN_PARAM(Type, Namespace, Name) Type Namespace::_##Name;
#endif
