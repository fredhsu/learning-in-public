# Calculus

## Hessian Matrix

A square matrix similar to the Jacobian, but with second order partial derivatives. It measures the curvature of the function locally.

$$
(\bold{H}_f)_{i,j}  = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

## Autodiff

Automatic differentiation (autodiff) is a method for efficiently computing derivatives of functions expressed as computer programs. Every program can be broken down into elementary arithmetic operations (add, multiply, divide, etc.) and common functions (exp, log, sin, cos, …). By representing these operations in a computational graph and applying the chain rule, autodiff systematically propagates derivatives through the graph.
Unlike symbolic differentiation (which can produce unwieldy expressions) or numerical differentiation (which suffers from approximation error), autodiff provides exact derivatives up to machine precision.
There are two main modes:

- Forward mode: propagates derivatives from inputs to outputs.

- Reverse mode ([backpropagation]): propagates derivatives from outputs to inputs, and is particularly efficient when computing gradients of scalar outputs with respect to many inputs (as in machine learning).

## Backpropagation

An efficient way to calculate the gradients of the nodes in the computational graph. It works by calculating the partial derivatives of each node using the chain rule, starting from the loss node and working backward from the output to the parameters.

Example:

$$
\frac{\partial L}{\partial \theta_i} =
\frac{\partial L}{\partial f_k} \cdot
\frac{\partial f_k}{\partial f_{k-1}} \cdot
\dots
\frac{\partial f_{i+1}}{\partial f_i} \cdot
\frac{\partial f_{i}}{\partial \theta_i}
$$

Backpropagation is a special case of reverse-mode [[autodiff]], using algorithmic as opposed to symbol methods to calculate the gradients.

## Jacobian Matrix

A Jacobian matrix is a collection of all the first-order partial derivatives of a vector valued function, $f=\mathbb{R}^n \to \mathbb{R}^m$

$$
J = \nabla_x f = \frac{df(x)}{dx} =
\begin{bmatrix}
\frac{\partial f(x)}{\partial x_1} \ldots
\frac{\partial f(x)}{\partial x_n}
\end{bmatrix} =
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

where

$$
f(x) =
\begin{bmatrix}
f_1(x) \\
f_2(x) \\
\vdots \\
f_m(x)
\end{bmatrix},
\quad
x =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

### Applications

- The determinant of the Jacobian (when $m=n$) gives the local volume scaling factor under the transformation.
- Geometrically the Jacobian is the scaling factor when we transform an area/volume.

## Taylor Polynomial

Provides an approximation of a function that gets more accurate as $n$ increases.

$$T_n(x) := \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k$$
where $f^{(k)}(x_0)$ is the kth derivative of f at $x_0$

Taylor series is the Taylor polynomial with $n=\infty$

## Gradient

A column vector with partial derivatives with regard to the elements of the original vector.

The gradient of a scalar-valued function $f:\mathbb{R}^n \to \mathbb{R}$ with respect to the vector $x = (x_1,\dots,x_n)$ is the vector of partial derivatives:

$$

\nabla_x f(x) =
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
\in \mathbb{R}^n

$$

It points in the direction of the steepest increase of $f$.

## Chain rule for partial derivatives

$$\frac{\partial}{\partial x}(g \circ f)(x) = \frac{\partial}{\partial x}(g(f(x))) = \frac{\partial g}{\partial f} \frac{\partial f}{\partial x}$$

This can be applied to parameterized functions to get the [[gradient]], for instance $f(x_1,x_2)$ where $x_1$ & $x_2$ are functions of $t$, we can apply the chain rule to get the gradient.

$$

\frac{d}{dt} f(x_1(t), x_2(t))
= \frac{\partial f}{\partial x_1}\frac{dx_1}{dt}

- \frac{\partial f}{\partial x_2}\frac{dx_2}{dt}.
$$

For multivariable functions use [[Jacobian Matrix]]
Let $f:\mathbb{R}^n \to \mathbb{R}^m$ and $g:\mathbb{R}^m \to \mathbb{R}^p$.  
Then:

$$
D(g \circ f)(x) = Dg(f(x)) \cdot Df(x),
$$

where:

- $Df(x)$ is the $m \times n$ Jacobian of $f$,
- $Dg(f(x))$ is the $p \times m$ Jacobian of $g$,

The result is a $p \times n$ Jacobian.
