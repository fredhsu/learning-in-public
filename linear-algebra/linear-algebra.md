# Linear Algebra notes

## Summary

Linear algebra provides tools for solving and manipulating vectors and matrices. Beginning with vectors that define a length and direction, represented by a list of numbers, and more abstractly follow rules for vector addition and multiplying by a scalar.
Vectors can be combined using linear combinations, essentially adding and scaling vectors to create a new vector. and can be used to represent a linear equation.
From a networking perspective
from a AI/ML perspective

## Vectors

Vectors are a fundamental concept in linear algebra and can have multiple interpretations from physics, to computer science, and mathematics. For our purposes we will use assume they represent a line that begins at the origin, and goes to the point represented by the vector.
They are represented[^1] like this:

$\begin{bmatrix} 1 \\ 2 \end{bmatrix}$

The primary vector operations are:

- vector addition - combining vectors end to end
- multiplication by a scalar - scales the vector

A vector exists outside of its coordinates and is not the same as its coordinates
Vectors can represent different coordinates in different coordinate spaces through [[Linear Transformation]].

### Orthogonal Vectors

Vector that is perpendicular to another vector or vector space. It has a dot product of zero with the other vector, or with all the vectors of the other vector space.

### Orthonormal Vectors

[[Orthogonal Vectors]] with length 1.

### Basis Vector

Basis vectors of a [[Vector Space]] are a set of [[Linearly Independent]] vectors that span the full space.
Since a vector space can have infinitely many vectors, using the basis allows us to succinctly define and work with the VS.

Given S subset of V, S is basis of V if

- S is LI
- Span(S) = V
- The elements of S are basis vectors
- MoML pg 57

### Orthonormal Basis (ONB)

Basis vectors that are orthonormal

- [[inner product]] is 0 : <b_i, b_j> = 0 for i != j (orthogonal),
- <b_i, b_i> = 1 ; have length = 1

Can be found by using [[Gram-Schmidt algorithm]]

### Gram-Schmidt algorithm

An iterative method for constructing an [[Orthonormal Basis]] from a set of [[Linearly Independent]] vectors. It works by orthogonalizing each vector with respect to the previous ones, then normalizing each resulting vector to have unit length.[^gsjupyter]

### Linear Combination

A combination of scaling (multiplying) or adding vectors

### Norm

The magnitude of a vector from the origin.
Common norms are:

- L1 (Manhattan): sum of absolute values
- L2 (Euclidean): square root of a sum of squares.

Norms have three properties:

1. Non-negative
2. Homogeneity: ||ax|| = ||a|| ||x|| for any scalar a
3. Triangle inequality = ||x+y|| <= ||x||+||y||

Used in: Regularization, Optimization (Grad Descent), Loss functions, distance metrics, batch and layer normalization.

### Vector Space

The vector space defines a set V of vectors, a field F, vector addition, and scalar multiplication that follow a vector space axioms such as closure, associativity, identity elements, inverses, and distributive properties.

For example, ℝ³ is a vector space.

Can be described by the [[Basis Vector]]

### Linearly Dependent

A subset of vectors contains the zero vector or one of its vectors can be represented as a [[Linear Combination]] of other vectors.
This implies there is a "redundant" vector in the set.
Another definition would be if the null vector 0 can be obtained through linear combination. [^moml56]

### Linearly Independent

No vectors in a set can be written as linear combinations other vectors in the set.
Can be found by Gaussian Elimination and checking if there are no non-zero rows, calculating the determinant for a square matrix and checking if it is != 0, or if rank = # of vectors.
If adding another vector increases the [[Span]] they are linearly independent.

Another definition would be it is linearly independent iff when the sum of all vectors multiplied by coeffiencts is zero, all coefficients are zero[^moml56]

### Span

The span is all the vectors that can be reached by using a linear combination of a given set of vectors. Adding linearly dependent vectors does not increase the span.

### Inner Product

- ⟨u,v⟩→R
- A bilinear mapping that is symmetric and positive definite (always positive when applied to itself).
- Takes two vectors and returns a scalar. $\langle x,y \rangle = x_1 y_1 + x_2 y_2$
- Measures length[^3] and angle[^4] between them. If ⟨u,v⟩=0, then u⊥v, if ||x|| = 1 = ||y|| it is also orthonormal.
- $\langle x,y \rangle = cos \| x \| \|y\| \alpha$
- Generalization of [[Dot Product]]
- MML pg 73-76

## Matrices

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

Any matrix can be geometrically viewed as a [[Linear Transformation]]. In fact, you can create a 1:1 correspondence between a set of linear transformations $f$ and a matrix $A$ for a given basis. From this perspective a matrix is simply a representation of linear transformations relative to a given basis.

### Orthogonal Matrix

- Has columns that are [[Orthonormal]], resulting in A^(-1) = A^T, which makes calculating the inverse efficient
- Results in a rotation when viewed as a linear transformation, the transformation is done relative to an [[Orthonormal Basis]]

### Linear Transformation

By multiplying a vector by a matrix, we can perform transformations of the vector. So geometrically a matrix represents a linear transform of a vector.

Linear Transformations have the key properties:

1. Lines remain lines, they don't become curves
2. Origin stays the same
3. Grid lines stay equally spaced and parallel

The transformation can be represented by a matrix describes of how the basis vector changes from the domain to the codomain. This creates a 1:1 correspondence between the transformation functions and the matrix which represents it. The values of the matrix are the images of the domain in the codomain. It can be compared to the allegory of caves, where the transformation is the reality and the matrix is the shadow representation of the transformation.

**Definition (Matrix of a Linear Transformation)**

Let $V$ and $W$ be finite-dimensional vector spaces over a field $\mathbb{F}$ with ordered bases
$\mathcal{B} = \{v_1, v_2, \ldots, v_n\}$for $V$ and
$\mathcal{C} = \{w_1, w_2, \ldots, w_m\}$for $W$ .

Let $T: V \to W$ be a linear transformation. For each basis vector $v*j \in \mathcal{B}$, there exist unique scalars
$a*{1j}, a*{2j}, \ldots, a*{mj} \in \mathbb{F}$ such that

$$
T(v_j) = \sum_{i=1}^{m} a_{ij} w_i.
$$

The matrix of $T$ with respect to the bases $\mathcal{B}$ and $\mathcal{C}$ is the $m \times n$ matrix

$$
[T]_{\mathcal{C} \leftarrow \mathcal{B}} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\\\
a_{21} & a_{22} & \cdots & a_{2n} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}.
$$

Each **column** of this matrix is the coordinate vector of $T(v_j)$ expressed in the basis $\mathcal{C}$.

- [Interactive Tool](https://claude.ai/public/artifacts/ebfef9fb-c08b-48ca-a9ed-9ec68ef6ba6b)
- [3 Blue 1 Brown video](https://www.youtube.com/watch?v=kYB8IZa5AuE)

### Determinant

For a 2-D vector space, it gives the change in the area of a square created by the [[Basis Vector]]s. Change in volume for a cube in 3D.
When det = 0, it squishes the area down to a line or point, and indicates the vectors of the matrix are [[Linearly Dependent]].
If det is negative, there is a "flip", but the change in area is equivalent to the absolute value of the determinant.
[^3b1bdet]

The determinant can only be found for a square matrix.

Calculating the determinant
For 2D
For 3D
For nxn - reduce to 2x2

### Trace

The diagonal sum of the matrix

### Injective

A matrix is injective (one-to-one) if every input maps to a distinct output. This does not necessarily mean that all values in the co-domain are covered. In other words, every input gets mapped to one output, with no overlaps. If you know that x -> y, then if something else maps to y it must be equal to x. Injective also means $ker(A) = \{0\}$ the Kernel is equal to the zero vector. A matrix can only be injective if it is square or "tall" (m >= n). More formally:

$f: A \rightarrow B$ is injective if:

$$
f(x_1) = f(x_2) \Rightarrow x_1 = x_2
$$

$$
x_1 \neq x_2 \Rightarrow f(x_1) \neq f(x_2)
$$

### Surjective

A matrix is surjective (onto) if every vector in the co-domain has a mapping from the domain. This could mean multiple inputs map to the same output, but every output must have a mapping from the input space. A matrix can only be surjective if it is square[^surjinjsquare] or "wide" (n <= m).
This means $Ax=b$ has a solution for all $b \in R^m$ if $A$ is surjective.[^surj]
Formally:

$$
\forall y \in B, \exists x \in A \text{ such that } f(x) = y
$$

### Bijective

A matrix is bijective if it is both [[Injective]] and [[Surjective]]. A bijective mapping has an inverse, so it defines an [[invertable matrix]] and must be square. A bijective matrix preserves the structure during transformation allowing "change of variable" transformations.

### Summary of Injective/Surjective/Bijective

| Shape       | Transformation        | Injective?        | Surjective?           |
| ----------- | --------------------- | ----------------- | --------------------- |
| **2D → 2D** | Shear/rotate          | ✅ if no collapse | ✅ if spans output    |
| **2D → 3D** | Plane floating in 3D  | ✅                | ❌                    |
| **3D → 2D** | Flatten cube to plane | ❌                | ✅ if plane is filled |

---

| Matrix Shape       | Injective?  | Surjective? | Bijective? |
| ------------------ | ----------- | ----------- | ---------- |
| **Square (n = m)** | ✅ Yes      | ✅ Yes      | ✅ Yes     |
| **Tall (m > n)**   | ✅ Possible | ❌ No       | ❌ No      |
| **Wide (m < n)**   | ❌ No       | ✅ Possible | ❌ No      |

### Invertible Matrix

For a matrix to be invertible, there must be a mapping from each of vectors in the domain to the image of its inverse, i.e. it is [[Bijective]]. This means no non-zero vectors will be mapped to zero, only zero will map to zero.[^noninvertible] Invertible matrices allow stretching, rotating, etc. but preserves the dimensionality. The inverse is defined by: $AA^{-1} = I$

### Characteristic Polynomial

For a square matrix:
$$P_a(\lambda) = det(A - \lambda I)$$
Used in calculating [[Eigenvector]]

### Eigenvectors and Eigenvalues

Eigenvectors and values characterize the [[Linear Transformation]] of a square matrix. The mathematical relationship is:
$$Av = \lambda v$$
where v is the eigenvector and $\lambda$ is the eigenvalue.

[Visualization of Eigenvector and value](https://claude.ai/public/artifacts/bc712e4f-70d5-4dba-8ff0-5d79695343f4)

#### Eigenvector

Eigenvectors point in the directions that are preserved (or exactly reversed) by a linear transformation. While most vectors change both magnitude and direction when transformed, eigenvectors only change in magnitude by a factor equal to their corresponding eigenvalue.

#### Eigenvalue

Eigenvalues indicate how much the eigenvectors are stretched as a result of the linear transformation. If the eigenvalue is 1 then there is no change, 0 means the eigenvector becomes the zero vector and reduces the dimensionality of the vector space. The number of zero eigenvalues indicates how much the dimensionality of the vector space is reduced.

#### Calculation

For a matrix A, eigenvector x, and eigenvalue $\lambda$ : $Ax = \lambda x$

- $det(A - \lambda I_n = 0)$
- If $A \in R^{nxn}$ is symmetric, there is an [[ONB]] of the vector space with the eigenvector of A and a real eigenvalue
- The [[Determinant]] of a matrix is equal to the product of its eigenvalues: $det(A) = \Pi i=1$ to $n \lambda_i$
  - Ties in the fact that the determinant calculates the area of the transformation with the eigenvalues.
- Solving for the [[Eigenvalue]] and [[Eigenvector]]
  1. Set [[Charateristic Polynomial]] = 0: $$P_A(\lambda) = 0$$
  2.

### PageRank

Uses the [[Eigenvector]] of the maximal [[Eigenvalue]]s to rank a page based on the incoming links and how important they are.

## Null space

Has dimension equal to the number of zero [[Eigenvalue]]s.

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

[^3b1bdet]: Great video from 3b1b on this.

[^3]: Distance between vectors for an inner product space (V, <.,.>) : d(x,y) := |x - y| = sqrt(<x - y>, <x - y>). Relates to [[L2 Norm]].

[^4]: cos w = <x, y> / (||x|| ||y||) by Cauchy-Schwartz Inequality

[^moml56]: Theorem 2 from MoML

[^gsjupyter]: See Jupyter notebook for an implementation

[^noninvertible]: Conversely, if the matrix is non-invertible, then it will collapse some part of its space to a lower dimension by mapping a non-zero vector to zero.

[^surj]: Because everything in $b$ can be mapped to by A from something in $x$

[^surjinjsquare]:
    If a matrix is square and surjective, it is also injective because the number of columns of a square matrix is equal to the dimension of the domain (full rank) making the kernel trivial which is the definition of injective. This also makes it [[Bijective]] and therefore an [[Invertible Matrix]]

    $$
    $$
