## Coding style
We chose to abide by the following coding style rules.

1. Names representing classes must be in camel case:
  `MyClass`
2. Variable and method names must be in lower case, using underscores to separate words:
  `my_variable`, `my_method()`
3. Names of protected and private members must start with an underscore:
  `_my_private_member`, `_my_private_method()`
4. File names must be in lower case, using underscores to separate words.
  A file which contains a class `MyClass` should be named `my_class.hpp`
5. File structure mirrors namespace structure.
  For instance `gen::MyClass` is in the file `gen/my_class.hpp`
6. Named constants (including enumeration values) must be placed in the `cst` namespace within the current namespace
  ```
  namespace cst {
    static constexpr int a_number = 3529;
  }
  ```
7. Getters should have the name of the attribute:
  `this->_objs` should be accessed using `this->objs()`
8. Setters should start with "set\_" followed by the name of the attribute:
  `set_objs(const std::vector& ov)`
9. The public section should be the first section of a class
10. Type names defined using typedefs/aliases should end with "\_t": `iterator_t`

## Code formatting
We also follow the coding style rules enforced by `clang-format` and a custom configuration. See the [format_code](https://github.com/resibots/format_code) repository and software to follow this standard.
