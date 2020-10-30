---
author:
- 'Victoria DiTomasso, Martha Obasi, Matt Egan, Thomas Shay Hill'
title: |
  **Designing Software for Automatic Differentiation\
  Milestone 1**
---

Introduction {#sec:Intro}
============

Derivatives play an important role in a wide variety of scientific
computing applications including sensitivity analysis, optimization, 3D
modeling, systems of nonlinear equations and inverse problems
(Karczmarczuk 2001). Our software package implements forward mode
automatic differentiation (AD) in order to compute the derivative of any
differentiable function $f(x)$ for any $x \in R$. Our package takes as
inputs any differentiable function $f(x)$ and any point $x \in R$ and
returns the derivative at point $x$.

Background {#sec:Background}
==========

The efficient and accurate calculation of derivatives is an integral
task in many areas of science and engineering. Automatic differentiation
is just one of several methods for calculating the derivative of a
function computationally: the other most common methods are symbolic
differentiation and numerical differentiation. Automatic differentiation
offers distinct advantages over these other approaches.

Symbolic differentiation re-works mathematical expressions in order to
arrive at the derivative of $f(x)$. While the result is indeed the true
derivative, symbolic methods can suffer from long compute times. By
contrast, numerical methods for calculating the derivative - such as the
finite-difference approach - rely on calculating the change in $f(x)$
over a minute but finite interval $h$. Numerical methods tend to be both
efficient and intuitive, but are inaccurate for both large and small
values of $h$. For large values of $h$, the interval is too coarse to
yield an accurate representation of a function's rate of change.
Conversely, as $h \rightarrow 0$, error increases as a result of
truncation error: this approach in other words suffers from numerical
instability. Thus numerical methods are necessarily approximations, and
are incapable of calculating the true derivative (Hoffman 2016).

Automatic differentiation suffers from neither of these issues. Also
known as \"algorithmic differentiation\", automatic differentiation
treats a given function $f(x)$ as a composite function consisting of a
nested set of discrete operations which cannot be further subdivided.
These discrete operations are members of a finite set of so-called
\"elementary functions\", whose derivatives have been pre-programmed.
Automatic differentiation breaks the function $f(x)$ into its
constituent, nested elementary operations; and then computes the
derivative of each of these sequentially, using the Chain Rule of
Calculus (Griewank 2003).

Forward mode automatic differentiation combines partial derivatives
starting with the input variables and moving forwards to the output
variables. We can derive derivative formulae for any function
constructed from elementary functions and operations using a
computational graph and trace table as shown below. Take for example the
derivative for $f(x) = tan (x^2)$. If doing this by hand, first we would
draw the simple computational plot decomposing the equation into
elementary functions and operations as shown below:

Starting from the node, we then move forward through the graph computing
the derivative of the next node with respect to $x$ as shown in the
following trace table:

  Trace   Elementary Operation   Derivative of Elementary Function
  ------- ---------------------- -----------------------------------
  $x_1$   $x$                    $\dot{x_1}$
  $v_1$   $x^2$                  $2x$
  $f$     $tan(v_1)$             $sec^2(v_1)\dot{v_1}$

From the table, we have $v_1 = x^2$ and $\dot{v_1} = 2x$ therefore, the
derivative can be expressed as $2x*sec^2(x^2)$. The forward mode is
essentially an implementation of this method of calculating derivatives.

How to Use {#sec:HowTo}
==========

The framework works around the mvmtFloat object. To evaluate a function
for a value, you need to create this value as a mvmtFloat. Simple
operations like + - \* / are overloaded, so you can just do:

``` {.python language="Python"}
from AD import mvmtFloat

    #a = mvmtFloat(point to evaluate array, seed vector))
    a = mvmtFloat(3, np.array([1,0]))

    print('a=', a)
    print('a+1=', a+1)
    print('2*a=', 2*a )
    print('a**2=', a**2 )

    > a      =  mvmtFloat with Value: 3, Derivative: [1 0]
    > a + 1  =  mvmtFloat with Value: 4, Derivative: [1 0]
    > 2*a    =  mvmtFloat with Value: 6, Derivative: [2 0]
    > a**2   =  mvmtFloat with Value: 9, Derivative: [6 0]
```

Alternatively, you can do as follows:

``` {.python language="Python"}
from AD import mvmtFloat
    
    a = mvmtFloat(3, np.array([1,0]))
    f = lambda a: a**2 
    v = f(x)
    
    print(v.derivative)
    > [6 0]
    print(v.value)
    > 9
```

More complex functions can be evaluated in a similar manner returning
partials with respect to the select variables. Take the following
example:

``` {.python language="Python"}
from AD import mvmtFloat
    
    a = mvmtFloat(np.array(3), np.array([1,0]))
    b = mvmtFloat(np.array(7), np.array([0,1]))
    
    print('a**2*b=', a**2*b)
    > a**2*b = mvmtFloat with Value: 63, Derivative: [42 9]
```

Software Organization {#sec:Organization}
=====================

The MVMT107/cs107-FinalProject repository is organized as follows:

This package consists of one primary module, AD. This module contains
the constructors and functions necessary to execute forward mode
automatic differentiation. The test suite (AD/tests/) contains both unit
and integration tests. This package is set up with TravisCI, and so the
success of the tests is updated in real time and represented by the
TravisCI build badge in the README. The coverage of the test suite is
also continuously updated and displayed by the Codecov badge, located in
the README.

This package can be installed from source by first downloading the
repository, either from the GitHub website, GitHub Desktop app or using
the [`git clone`]{style="background-color: light-gray"} command.

[`git clone https://github.com/MVMT107/cs107-FinalProject.git`]{style="background-color: light-gray"}

Once in your local copy of the repository, run install setup.py using
Python from the command line. This package is not yet indexed with PyPI,
and as such, is not pip-installable. In order to maintain flexibility in
its development, and due to the simplicity of the package, it has not
been packaged using a framework.

[`python setup.py install`]{style="background-color: light-gray"}

Implementation {#sec:Implementation}
==============

Automated Differentiation will be implemented using operator overloading
and function overriding.

A new class, named mvmtFloat, will be defined with instance attributes
that contain the value and derivative for a variable at a given point.
This class will have its Arithmetic, In-place, and Comparison operators
overridden.

The functionality of the overridden Arithmetic and In-Place methods will
be similar to the standard operators, but will implement the correct
calculus derivative rule when one or both arguments are of this class.
The return values of these methods will be instances of the mvmtFloat
class. The overridden Comparison operators will implement the standard
comparison logic on the value attribute.

The python math and numpy modules will be imported. The elementary
functions from the math module will be overridden, similar to the
Arithmetic operators.

Additionally, separate functions for calculating partial derivatives and
jacobians will be provided:

-   A partials() function will accept as input: a vector of instances of
    the mvmtFloat class as well as a function that accepts multiple
    variables. The partials() function will return the input vector of
    instances where the derivative attribute of each has been set to the
    partial derivative of the input function with respect to that
    variable. The partial derivative calculation will be implemented by
    setting the derivative attribute of each instance to gradient
    vector, as numpy.array().

-   A jacobian() function will accept as input a vector of instances of
    the mvmtFloat class as well as a vector of functions. This function
    will return a jacobian matrix as a numpy.array().

10

Griewank, Andreas. \"A mathematical view of automatic differentiation.\"
, no. 1 (2003): 321-398.

Hoffman, Phillip H. "A Hitchhiker's Guide to Automatic Differentiation."
:775-811, 2016.

Karczmarczuk, Jerzy. "Functional Differentiation of Computer Programs\".
, 14, 35--57, 2001.
