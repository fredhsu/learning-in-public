# Linear Algebra notes

## Summary

Linear algebra provides tools for solving and manipulating vectors and matrices. Beginning with vectors that define a length and direction, represented by a list of numbers, and more abstractly follow rules for vector addition and multiplying by a scalar.
Vectors can be combined using linear combinations, essentially adding and scaling vectors to create a new vector. and can be used to represent a linear equation.
From a networking perspective
from a AI/ML perspective

## Vector

Vectors are a fundamental concept in linear algebra and can have multiple interpretations from physics, to computer science, and mathematics. For our purposes we will use assume they represent a line that begins at the origin, and goes to the point represented by the vector.
They are represented[^1] like this:

$\begin{bmatrix} 1 \\ 2 \end{bmatrix}$

The primary vector operations are:

- vector addition - combining vectors end to end
- multiplication by a scalar - scales the vector

A vector exists outside of its coordinates and is not the same as its coordinates
Vectors can represent different coordinates in different coordinate spaces through [[linear transformation]].

## Basis Vectors

Basis vectors of a [[Vector Space]] are a set of [[Linearly Independent]] vectors that span the full space.
Since a vector space can have infinitely many vectors, using the basis allows us to succinctly define and work with the VS.

Given S subset of V, S is basis of V if

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

## Linear Transformation

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

Best described by the [[Basis]]

## Span

The span is all the vectors that can be reached by using a linear combination of a given set of vectors. Adding linearly dependent vectors does not increase the span.

## Linear Dependence

A subset of vectors contains the zero vector or one of its vectors can be represented as a [[linear combination]] of other vectors.
This implies there is a "redundant" vector in the set.
Another definition would be if the null vector 0 can be obtained through linear combination. [^moml56]

## Linear Independence

No vectors in a set can be written as linear combinations other vectors in the set.
Can be found by Gaussian Elimination and checking if there are no non-zero rows, calculating the determinant for a square matrix and checking if it is != 0, or if rank = # of vectors.
If adding another vector increases the [[Span]] they are linearly independent.
Another definition would be it is linearly independent iff when the sum of all vectors multiplied by coeffiencts is zero, all coefficients are zero[^moml56]

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

The determinant can only be found for a square matrix.

Calculating the determinant
For 2D
For 3D
For nxn - reduce to 2x2

## Trace

The diagonal sum of the matrix

## Characteristic Polynomial

For a square matrix:
$$P_a(\lambda) = det(A - \lambda I)$$
Used in calculating [[Eigenvectors]]

## Eigenvectors and Eigenvalues

- Eigenvectors and values characterize the linear mapping of a square matrix.

### Eigenvectors

- Eigenvectors point in the direction of the mapping

### Eigenvalues

- Eigenvalues indicate how much the eigenvectors are stretched

### Calculations

- For a matrix A, eigenvector x, and eigenvalue $$\lambda$$ : $$ Ax = \lambda x $$
- $$det(A - \lambda I_n = 0)$$
- If $A \elem R^(nxn)$ is symmetric, there is an [[ONB]] of the vector space with the eigenvector of A and a real eigenvalue
- The [[Determinant]] of a matrix is equal to the product of its eigenvalues: $$det(A) = \Pi i=1 to n \lambda_i$$
  - Ties in the fact that the determinant calculates the area of the transformation with the eigenvalues.
- Solving for the [[Eigenvalues]] and [[Eigenvectors]]
  1. Set [[Charateristic Polynomial]] = 0: $$P_A(\lambda) = 0$$
  2.

### PageRank

- Uses the [[Eigenvector]] of the maximal [[Eigenvalues]] to rank a page based on the incoming links and how important they are.

## Matrix Decomposition

Matrix decomposition breaks a matrix down into multiple factors, much like factoring an equation. The resulting
components can describe characteristics of the matrix, as well as make some calculations more efficient.

### Cholesky Decomposition

Decomposes into a lower triangular matrix and its transpose:
$A = LL^T$

$\begin{bmatrix} A_{00} A_{01} A_{02} \\ A_{10} A_{11} A_{12} \\ A_{20} A_{21} A_{22} \end{bmatrix}=$
$\begin{bmatrix} L_{00} L_{01} L_{02} \\ L_{10} L_{11} L_{12} \\ 0 0 0 \end{bmatrix} \begin{bmatrix} L_{00} L_{01} L_{02} \\ L_{10} L_{11} L_{12} \\ L_{20} L_{21} L_{22} \end{bmatrix}$
Example:

### Eigendecomposition

Breaks a matrix down into $PDP^(-1)$ but only works on square matrices.

### Singular Value Decomposition

Can decompose any matrix into $U \Sigma V^T$

## References

Math for Machine Learning (MML)
Math of Machine Learning (MoML)
3b1b

[^1]: I am using Pandoc and LaTeX to render some of this math. In this case pmatrix, bmatrix Bmatrix give parenthesis, brackets, braces matrix styles are useful latex options.

[^3b1b-det]: Great video from 3b1b on this.

[^3]: Distance between vectors for an inner product space (V, <.,.>) : d(x,y) := |x - y| = sqrt(<x - y>, <x - y>). Relates to [[L2 Norm]].

[^4]: cos w = <x, y> / (||x|| ||y||) by Cauchy-Schwartz Inequality

[^moml56]: Theorem 2 from MoML
