# Linear Algebra notes

## Vector

Vectors are a fundamental concept in linear algebra and can have multiple interpretations from physics, to computer science, and mathematics. For our purposes we will use assume they represent a line that begins at the origin, and goes to the point represented by the vector.
They are represented[^1] like this:

$\begin{bmatrix} 1 \\ 2 \end{bmatrix}$

The primary vector operations are:

- vector addition - combining vectors end to end
- multiplication by a scalar - scales the vector

## Matrix

A matrix can be used to represent a series of linear equations. For example given the following linear equations:
$$
x + 2y = 5
3x + 4y = 11
$$

This becomes:
$$
\begin{bmatrix}
1 & 2 & 5 \\
3 & 4 & 11 \\
\end{bmatrix}
$$

## Linear transformations

By multiplying a vector by a matrix, we can perform transformations of the vector. So geometrically a matrix represents a linear transform of a vector.

Linear Transformations have the key properties:

1. Lines remain lines, they don't become curves
2. Origin stays the same
3. Grid lines stay equally spaced and parallel

- [Interactive Tool](https://claude.ai/public/artifacts/ebfef9fb-c08b-48ca-a9ed-9ec68ef6ba6b)
- [3 Blue 1 Brown video](https://www.youtube.com/watch?v=kYB8IZa5AuE)

## Vector Space

The vector space defines a set V of vectors, a field F, vector addition, and scalar multiplication that follow a vector space axioms such as closure, associativity, identity elements, inverses, and distributive properties.

For example, ℝ³ is a vector space.

## Span

The span is all the vectors that can be reached by using a linear combination of a given set of vectors. Adding linearly dependent vectors does not increase the span.

## Linear Independence

No vectors in a set can be written as linear combinations other vectors in the set.
Can be found by Gaussian Elimination and checking if there are no non-zero rows, calculating the determinant for a square matrix and checking if it is != 0, or if rank = # of vectors.

## Norm

The magnitude of a vector from the origin.
Common norms are:

- L1 (Manhattan): sum of absolute values  
- L2 (Euclidean): square root of a sum of squares.

Norms have three properties:

1. Non-negative
2. Homogeneity: ||ax|| = ||a|| ||x|| for any scalar a
3. Triangle inequality = ||x+y|| <= ||x||+||y||

Used in: Regularization, Optimization (Grad Descent), Loss functions, distance metrics, batch and layer normalization.

[^1]: I am using Pandoc and LaTeX to render some of this math. In this case pmatrix, bmatrix Bmatrix give parenthesis, brackets, braces matrix styles are useful latex options.
