.. _dsl_control_flow:
.. |DC|        replace:: dynamic compilation
.. |IR|        replace:: intermediate representation (IR)
.. |DSL|       replace:: CuTe DSL
.. |Constexpr| replace:: **Constexpr** (compile-time Python value)

Control Flow
==================


Overview
--------
|DSL| walks Python's AST and converts each control-flow construct it finds into
structured |IR|.  You can therefore write ordinary Python loops and branches
while the compiler decides—statement by statement—whether to

* **evaluate at compile time** if it's a native Python control flow, or
* **emit intermediate representation (IR)** when the control flow is marked as dynamic.

Passing |IR| values to a native Python control flow will result in an error.


For Loops
---------
|DSL| recognises three kinds of ranges for ``for`` loops:

* ``range`` – the Python built-in, always lowered to |IR|
* ``cutlass.range`` - Same as Python built-in ``range``, but supports advanced unrolling and pipelining control
* ``cutlass.range_constexpr`` – unrolled at compile time


range(...)/cutlass.range(...)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use when you *always* want a loop in the generated |IR|, even if the inputs
are Python values.

cutlass.range_constexpr(...)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Runs in the Python interpreter and is fully unrolled before code generation.
All loop indices must be |Constexpr|.


**Example:**

.. code-block:: python

    @cute.jit
    def control_flow_examples(bound: cutlass.Int32):
        n = 10

        # ✅ This loop is Python loop, evaluated at compile time.
        for i in cutlass.range_constexpr(n):
            cute.printf("%d\\n", i)

        # ✅ This loop is dynamic, even when bound is Python value.
        for i in range(n):
            cute.printf("%d\\n", i)

        # ❌ This loop bound is a dynamic value, not allowed in Python loop.
        # Should use `range` instead.
        for i in cutlass.range_constexpr(bound):
            cute.printf("%d\\n", i)

        # ✅ This loop is dynamic, emitted IR loop.
        for i in range(bound):
            cute.printf("%d\\n", i)

        # ✅ This loop is dynamic, emitted IR loop with unrolling
        for i in cutlass.range(bound, unroll=2):
            cute.printf("%d\\n", i)


If-Else Statements
------------------

Standard Python ``if``/``elif``/``else`` is supported.

* **Predicate without annotation** → lowered to |IR|.
* **Predicate annotated with `cutlass.const_expr`** → evaluated at compile time.

**Example:**

.. code-block:: python

    @cute.jit
    def main(const_var: cutlass.Constexpr, dynamic_var: cutlass.Int32):
        # ✅ This branch is Python branch, evaluated at compile time.
        if cutlass.const_expr(const_var):
            cute.printf("Const branch\\n")
        else:
            cute.printf("Const else\\n")

        # ✅ This branch is dynamic branch, emitted IR branch.
        if dynamic_var == 10:
            cute.printf("Dynamic True\\n")
        else:
            cute.printf("Dynamic False\\n")

        # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
        if cutlass.const_expr(dynamic_var == 10):
            cute.printf("Bound is 10\\n")


While Loops
-----------

Standard Python ``while`` is supported.

* **Condition without annotation** → lowered to |IR|.
* **Condition annotated with `cutlass.const_expr`** → evaluated at compile time.

**Example:**

.. code-block:: python

    @cute.jit
    def main(dynamic_var: cutlass.Int32):
        n = 0

        # ✅ This is Python while loop, evaluated at compile time.
        while cutlass.const_expr(n < 10):
            cute.printf("Const branch\\n")
            n += 1

        # ✅ This is dynamic while loop, emitted IR while loop.
        while dynamic_var == 10:
            cute.printf("Dynamic True\\n")
            n += 1

        # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
        while cutlass.const_expr(n < dynamic_var):
            n += 1


Compile-Time Metaprogramming
----------------------------

Mix compile-time constructs with normal |DSL| code to generate specialised
kernels without runtime overhead.  A compile-time flag can, for example, toggle
an optional **ReLU** epilogue:

.. code-block:: python

   @cute.kernel
   def gemm(..., do_relu: cutlass.Constexpr):
       # main GEMM work
       ...
       if cutlass.const_expr(do_relu):    # compile-time guard
           # ReLU code is emitted only when do_relu is True
           ...

.. code-block:: text

   gemm(..., False)   # ReLU is omitted from the generated |IR|
   gemm(..., True)    # ReLU is included


Limitations of Dynamic Control Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Early-exit ``break``, ``continue``, ``pass`` or raising exception from
  control flow body are not yet supported.
* Operations in the control flow body are traced only when tracing is active in
  that region.
* Values originating in control flow body are not available outside the control
  flow.
* Changing type of a variable in control flow body is not allowed.

**Example:**

.. code-block:: python

    @cute.jit
    def control_flow_negative_examples(predicate: cutlass.Boolean):
        n = 10

        # ❌ This loop is dynamic, early-exit isn't allowed.
        for i in range(n):
            if i == 5:
                break         # Early-exit

        if predicate:
            val = 10
            # ❌ return from control flow body is not allowed.
            return
            # ❌ Raising exception from control flow body is not allowed.
            raise ValueError("This is not allowed")
            # ❌ Using pass in control flow body is not allowed.
            pass

        # ❌ val is not available outside the dynamic if
        cute.printf("%d\\n", val)

        if predicate:
            # ❌ Changing type of a variable in control flow body is not allowed.
            n = 10.0


.. _limitations:

Limitations
====================


Overview
---------------------
CuTe DSL is an embedded domain-specific language within Python. It utilizes a subset of Python's
syntax to provide a streamlined programming experience. It is important to understand that CuTe DSL
does NOT implement the complete Python language semantics in its JIT compilation process.

Programming Model
---------------------

**Python Native Data Types**
    CuTe DSL supports Python data structures when used for "meta-programming,"
    but these structures cannot be treated as dynamic values modifiable at runtime.
    For instance, lists and dictionaries can be used to configure kernel parameters
    during compilation or serve as containers for dynamic values,
    but their structure and organization cannot be altered during kernel execution.

    - **Static Values:**
        - Evaluated during JIT compilation phase
        - Immutable after compilation completes
        - Most Python native types (lists, tuples, dictionaries) are processed as static values
        - Primarily utilized for "meta-programming" and configuration purposes
        - Example: Lists can contain dynamic values but their structure cannot
          be modified during kernel execution

    - **Dynamic Values:**
        - Evaluated during runtime execution
        - Modifiable during execution of JIT-compiled functions
        - Only a specific subset of Python types are supported as dynamic values
        - Primitive types are automatically converted when passed as function arguments:
        
          - ``int`` → ``Int32`` (may be updated to ``Int64`` in future releases)
          - ``bool`` → ``Bool``
          - ``float`` → ``Float32`` (may be updated to ``Float64`` in future releases)

    The JIT compiler processes Python native types analogously to C++ template parameters.
    The compiled code cannot manipulate dynamic values of composite types
    such as lists, tuples, or dictionaries.

    For example, following code doesn't work as traditional Python program inside JIT function.

    .. code:: python

        @cute.jit
        def foo(a: Float32, b: Float32, i: Int32, res: cute.Tensor):
            xs = [a, b]
            # indexing list with dynamic index is not supported in CuTe DSL:
            res[0] = xs[i]

            if i == 0:
                # This will alway append Float32(3.0) to the list regardless
                # of the runtime value of `i`
                xs.append(Float32(3.0))

            for i in range(10):
                # This only append one element to the list at compile-time
                # as loop doesn't unroll at compile-time
                xs.append(Float32(1.0))

**Python Function**
    The DSL currently does not implement support for return values from Python functions,
    although this capability is planned for future releases.

    Example:

    .. code:: python

        @cute.jit
        def foo():
            return 1  # Currently unsupported in CuTe DSL

**Expression or Statement with Dependent Type**
    CuTe DSL implements static typing and does not support dependent types.
    The type of each expression must be determinable during compile time,
    in contrast to standard Python which implements dynamic typing.

    Example illustrating functionality in Python that is not supported in the DSL:

    .. code:: python

        # Valid in standard Python, but unsupported in CuTe DSL
        max(int(1), float(2.0))  # => 2.0 : float
        max(int(3), float(2.0))  # => 3   : int

    In CuTe DSL, types are promoted. For example:

    .. code:: python

        @cute.jit
        def foo(a: Int32, b: Float32, res: cute.Tensor):
            res[0] = max(a, b)  # Type is automatically promoted to Float32

    Following code using inlined if-else expression with dependent types
    is not supported in CuTe DSL:

    .. code:: python

        @cute.jit
        def foo(cond: Boolean, a: Int32, b: Float32, res: cute.Tensor):
            res[0] = a if cond else b


**Control Flow**
    The DSL transforms Python control flow statements (``if``, ``for``, ``while``)
    during Abstract Syntax Tree (AST) processing into structured control flow in MLIR
    which has the same constraints as dependent types. For instance,
    changing type of a variable in loop body is not allowed.

    - Variables must be defined prior to the control flow statement
    - Type consistency must be maintained throughout the control flow statement
    - Don't support early exit or return from if-else statements

    Example illustrating functionality in Python that is not supported in the DSL:

    .. code:: python

        @cute.jit
        def foo():
            a = Int32(1)
            for i in range(10):
                a = Float32(2)  # Changing type inside loop-body is not allowed in the DSL


**Built-in Operators**
    The DSL transforms built-in operators like ``and``, ``or``, ``max``, ``min``, etc.
    into MLIR operations. They also follow the same constraints of dependent types.
    For instance, ``a and b`` requires ``a`` and ``b`` to be of the same type.


**Special Variables**
    The DSL treats ``_`` as a special variable that it's value is meant to be ignored.
    It is not allowed to read ``_`` in the DSL.

    Example illustrating functionality in Python that is not supported in the DSL:

    .. code:: python

        @cute.jit
        def foo():
            _ = 1
            print(_)  # This is not allowed in the DSL


**Object Oriented Programming**
    The DSL is implemented on top of Python and supports Python's object-oriented programming (OOP) features
    for meta-programming at compile-time.

    However, similar to other composed data types, the DSL provides limited support for OOP when objects
    contain dynamic values. It is strongly recommended to avoid passing dynamic values between member methods
    through class state in your code.

    The following example illustrates functionality in Python that is not supported in the DSL
    without implementing the ``DynamicExpression`` protocol:

    .. code:: python

        class Foo:
            def __init__(self, a: Int32):
                self.a = a

            def set_a(self, i: Int32):
                self.a = i

            def get_a(self):
                return self.a

        @cute.jit
        def foo(a: Int32, res: cute.Tensor):
            foo = Foo(a)
            for i in range(10):
                foo.set_a(i)

            # This fails to compile because `a` is assigned a local value defined within the for-loop body
            # and is not visible outside of the loop body
            res[0] = foo.get_a()

    The example above fails to compile because ``Foo.a`` is assigned a local value defined within the for-loop body,
    which is not visible outside the loop body.

    The CuTe DSL implements an internal mechanism that provides limited support for OOP patterns via protocol.
    As the DSL continues to evolve to support additional features, this mechanism is subject to change
    and is not recommended for direct use in users' code for better portability.


**CuTe Layout algebra in native Python**
    Entirety of CuTe Layout algebra operations and APIs require JIT compilation. These 
    functionalities are exclusively available within JIT-compiled functions and cannot be 
    accessed in standard Python execution environments.
    
    Additionally, there exists a restricted set of data types that can be passed as arguments 
    to JIT-compiled functions, which further constrains their usage in native Python contexts. 
    Only following CuTe algebra types are supported as JIT function arguments: ``Tensor``, ``Pointer``, 
    ``Shape``, ``Stride``, ``Coord`` and ``IntTuple``. For ``Stride``, we don't support ``ScacledBasis``
    from native Python Context. Unfortunately, in the first release, we don't support 
    passing ``Layout`` under native Python Context.


Suggestions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reliable and predictable results:

- Avoid dependent types in your code
- Implement explicit type conversion for dynamic values
- Clearly distinguish between static (compile-time) and dynamic (runtime) values
- Use type annotations as much as possible to help JIT compiler
  to identify type to avoid ambiguity


.. code:: python

    # Example demonstrating explicit typing
    alpha = 1.0  # Explicitly defined as float using `1.0` instead of `1`
                 #  or `float(1)`
    beta = 2.0   # Explicitly defined as float
    result = max(alpha, beta)  # Will correctly perform float comparison


Testing CuTe DSL Behavior
=========================

When you are unsure about a CuTe DSL feature (codegen, pointer math, shuffle width, copy/predicate
semantics, etc.), it is usually faster and safer to validate it with a *minimal* standalone
``cute.compile`` repro than to iterate inside a full kernel.

Recommended workflow (especially on remote GPUs like p1 / H100)
---------------------------------------------------------------

1. Write a tiny script on the GPU host (e.g. ``/tmp/cute_*_test.py``) that:

   - Defines a minimal ``@cute.jit`` wrapper (or a small class with a ``@cute.jit __call__``)
   - Uses a single ``@cute.kernel`` with a very small launch (often ``grid=[1,1,1]``, ``block=[1,1,1]``)
   - Returns results by writing to a small output tensor and printing it from Python.

2. Compile explicitly with fake tensors, then run with real torch tensors:

   - CuTe DSL JIT functions do not accept torch tensors directly; you must call a compiled function.
   - Use ``cute.runtime.make_fake_tensor`` to build a compile signature cheaply.
   - Use ``options=\"--enable-tvm-ffi\"`` (matching the rest of kestrel).

3. Run it via ``uv`` on the remote host:

   .. code-block:: bash

      ssh p1 "cd ~/code/kestrel && ~/.local/bin/uv run python /tmp/cute_my_test.py"

   Or use a heredoc to avoid quoting issues:

   .. code-block:: bash

      ssh p1 "cd ~/code/kestrel && ~/.local/bin/uv run python - <<'PY'
      # ... repro ...
      PY"

Notes / gotchas
~~~~~~~~~~~~~~~

- Many CuTe/MLIR-building APIs (e.g. layout algebra like ``cute.make_layout``) require being inside
  the JIT/compile context. If you see errors like "requires a Context", move that construction
  inside a ``@cute.jit`` function (or inside code that is executed during ``cute.compile``), not at
  Python module import time.
- Keep repros small and single-purpose: fewer threads, minimal shared memory, and one feature under test.


Predicates (``pred=``) and cp.async zfill
=========================================

CuTe ``pred=`` basics
---------------------

- The ``pred=`` argument to ``cute.copy`` expects a *CuTe tensor/fragment* of ``cutlass.Boolean``
  (shape-compatible with the copy), not a Python ``bool`` and not a scalar ``cutlass.Boolean``.
- A simple way to build a constant predicate is:

  .. code-block:: python

      pred_false = cute.make_fragment_like(dst, Boolean)
      pred_false.fill(False)

  where ``dst`` is a view with the same shape as the copy destination.

cp.async + ``pred=False`` implies zfill
---------------------------------------

For the non-bulk cp.async op (``cpasync.CopyG2SOp``), we validated on H100 (SM90) that:

- If ``pred`` is false for a given cp.async copy, the destination in shared memory is **zero-filled**
  (PTX "ZFILL" semantics), instead of requiring an explicit manual clear loop.
- This makes it possible to replace expensive clearing codepaths (e.g. ``fill_swizzled`` in a hot loop)
  with a predicated cp.async copy.

We also validated in isolation that using ``pred=False`` is safe even if the source pointer is a dummy
value (including a null pointer), because the copy does not dereference global memory when zfilling.

Practical implication
~~~~~~~~~~~~~~~~~~~~~

In paged-KV (or any path where some rows are out-of-bounds), you can keep the same cp.async tiled-copy
shape and simply issue the copy with ``pred=False`` for invalid rows to get zeros in smem:

.. code-block:: python

    if should_load_row:
        cute.copy(atom_async_copy, src, dst)
    else:
        cute.copy(atom_async_copy, src, dst, pred=pred_false)  # zfill
