# Linear Algebra week 1 notes

## Vector

Vectors are a fundamental concept in linear algebra and can have multiple interpretations from physics, to computer science, and mathematics. For our purposes we will use assume they represent a line that begins at the origin, and goes to the point represented by the vector.
They are represented[^1] like this:

$\begin{bmatrix} 1 \\ 2 \end{bmatrix}$

The primary vector operations are:

- vector addition - combining vectors end to end
- multiplication by a scalar - scales the vector

## Matrix

A matrix can be used to represent a series of linear equations. For example given the following linear equations:
x + 2y = 5
3x + 4y = 11

This becomes:
$$
\begin{bmatrix}
1 & 2 & 5 \\
3 & 4 & 11 \\
\end{bmatrix}
$$

## Linear transformations

By multiplying a vector by a matrix, we can perform transformations of the vector.
<https://claude.ai/public/artifacts/ebfef9fb-c08b-48ca-a9ed-9ec68ef6ba6b>
<https://www.overleaf.com/learn/latex/Matrices>

## Vector Space

The vector space represents all the possible vectors that exist given

## Span

[^1]: I am using Pandoc and LaTeX to render some of this math. In this case pmatrix, bmatrix Bmatrix give parenthesis, brackets, braces matrix styles are useful latex options.
