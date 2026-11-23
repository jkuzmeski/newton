# Digital Shoe development

Use PufferLib's refactoring guide below as the standing direction for
Digital Shoe. Apply it to this project's Python and Warp code alongside
[Newton's source guidelines](../../CODING_GUIDELINES.rst) and the
[repository instructions](../../AGENTS.md).

Preserve simulation results, determinism, performance, artifact compatibility,
and documented entry points. Keep validation at artifact and caller boundaries
so malformed inputs still report the appropriate errors. Use assertions for
internal invariants. Apply the C and CUDA rules where those languages are used.

Source: [PufferLib 5.0, SKILL_ISSUES.md](https://github.com/PufferAI/PufferLib/blob/ea7f989371270d2948f96728bda8aec5e7634a4c/SKILL_ISSUES.md),
reproduced under the MIT license below.


# This is a hand-written guide of how I style and refactor PufferLib
# It is equally useful for human and AI brogrammers. And for other projects.
# If your name is not John Carmack, you should probably read it before blindly
# shoveling it into whatever language model is currently replacing you.

# Objective
Reduce source code length without golfing while preserving behavior and performance. Preserve existing determinism. Discard biases against long files or functions.

# Joseph's Stupid Refactoring Algorithm
1. Inline every function that is only used once and tighten former call sites
2. Eliminate defensive checks. Replace complex error handling with plain asserts.
3. Reduce deeply nested code by merging and inverting conditionals
4. Co-optimize multi-consumer functions with their callers
5. Apply syntax and style guide as a final pass

# Syntax & Style (General)
- Do not split up code into more files
- Soft 80-col / hard 100-col limit
- Do not one-line loops or conditionals
- 4-space indents. Do NOT match opening parens
- Next line continuations indent 4 extra spaces instead of matching parens
- Apply semantic vertical spacing between blocks of code sparingly
- Tests are important but their length and code quality is not counted
- Do not add source complexity or shims for ease of testing
- Do not block off comments with --- or ### etc.

# C
- Do not use header files as lists of declarations. Treat them as source files.
- Avoid forward declarations. Define functions as close as possible to first use.
- Preallocate all memory at init by default and let the OS free it on close
- Runtime allocations should be rare and freed in the same scope
- Avoid keyword bloat, such as redundant static, const, inline, etc.
- Use macros only for constants and constant expressions, not for conditionals
- Use struct initializer syntax foo = {.a = 1, .b = 2} instead of setter wrappers
- Do not use additional scoping blocks, i.e. bare {}. Dedupe names instead.

# CUDA
- Treat CUDA files as CUDA C99
- Do not use the C++ standard library, templates, or classes
- Do not use the C++ versions of C language features like nullptr and static_cast
- Exception: existing/user-directed overloading for different numeric types
- Do not null params with (void) (Wunused-parameter is disabled)
- Avoid redundant casts where autocast will do (Wnarrowing is disabled)
- Follow the C style guide above

## PufferLib license

MIT License

Copyright (c) 2022 PufferAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
