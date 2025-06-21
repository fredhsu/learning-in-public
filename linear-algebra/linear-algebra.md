# Linear Algebra notes

## Vector

Vectors are a fundamental concept in linear algebra and can have multiple interpretations from physics, to computer science, and mathematics. For our purposes we will use assume they represent a line that begins at the origin, and goes to the point represented by the vector.
They are represented[^1] like this:

$\begin{bmatrix} 1 \\ 2 \end{bmatrix}$

The primary vector operations are:

- vector addition - combining vectors end to end
- multiplication by a scalar - scales the vector

## Basis Vectors

Basis vectors of a [[Vector Space]] is a set of [[Linearly Independent]] vectors that span the full space.

- Given S subset of V, S is basis of V if
- S is LI
- Span(S) = V
- The elements of S are basis vectors
- MoML pg 57

### Orthormal Basis (ONB)

Basis vectors that are orthonormal (orthogonal and have length 1)
- [[inner product]] is 0 : <b_i, b_j> = 0 for i != j (orthogonal), 
- <b_i, b_i> = 1 ; have length = 1

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

Any matrix can be geometrically viewed as a [[Linear Transformation]]

### Orthogonal Matrix

- Has columns that are [[orthnormal]], resulting in A^(-1) = A^T, which makes calculating the inverse efficient
- Results in a rotation when viewed as a linear transformation, the transformation is done relative to an [[orthonormal basis]]

## Linear Combination
A combination of scaling (multiplying) or adding vectors

## Linear Transformations

By multiplying a vector by a matrix, we can perform transformations of the vector. So geometrically a matrix represents a linear transform of a vector.

Linear Transformations have the key properties:

1. Lines remain lines, they don't become curves
2. Origin stays the same
3. Grid lines stay equally spaced and parallel

The transformation can be described in terms of how the basis vector changes.
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
If adding another vector increases the [[Span]] they are linearly independent.

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

## Inner Product

- ⟨u,v⟩→R
- A bilinear mapping that is symmetric and positive definite (always positive when applied to itself). 
- Takes two vectors and returns a scalar. 
- Measures length[^3] and angle[^4] between them. If ⟨u,v⟩=0, then u⊥v, if ||x|| = 1 = ||y|| it is also orthonormal.
- Generalization of [[Dot Product]]
- MML pg 73-76

## Determinant

For a 2-D vector space, it gives the change in the area of a square created by the [[basis vector]]s. Change in volume for a cube in 3D. 
When det = 0, it squishes the area down to a line or point, and indicates the matrix is [[linearly dependent]].
If det is negative, there is a "flip", but the change in area is equivalent to the absolute value of the determinant.
[^3b1b-det]


## References
Math for Machine Learning (MML)
Math of Machine Learning (MoML)
3b1b

[^1]: I am using Pandoc and LaTeX to render some of this math. In this case pmatrix, bmatrix Bmatrix give parenthesis, brackets, braces matrix styles are useful latex options.
[^3b1b-det]: Great video from 3b1b on this.
[^3]: Distance between vectors for an inner product space (V, <.,.>) : d(x,y) := |x - y| = sqrt(<x - y>, <x - y>). Relates to [[L2 Norm]].
[^4]: cos w = <x, y> / (||x|| ||y||) by Cauchy-Schwartz Inequality
