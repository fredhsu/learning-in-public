# Deep Learning

## Logistic Regression

- Logistic Regression is a **classification technique** used to predict binary outcomes (0 or 1).  
- It models the probability that a given input belongs to a particular class by applying the **logistic (sigmoid) function** to a linear combination of input features.
- Logistic regression is a **linear classifier**—it uses a linear decision boundary in the feature space.

### Sigmoid Activation Function

The sigmoid (logistic) function maps any real-valued number into the range (0, 1):

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where

$$
z = w^T x + b
$$

The output \( a = \sigma(z) \) represents the predicted probability \( \hat{y} \) that \( y = 1 \).

### Loss Function (Binary Cross-Entropy)

For a single training example, the **logistic regression loss** is defined as:

$$
L(\hat{y}, y) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
$$

where:

- \( y \) ∈ {0, 1} is the true label,  
- \( \hat{y} = \sigma(z) \) is the predicted probability.

### Cost Function

For a dataset with \( m \) training examples, the **cost function** (average loss) is:

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)}) =
-\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

This cost function measures how well the model’s predictions fit the actual data.

### Gradient of the Loss Function

To minimize the cost function using **gradient descent**, we compute the derivative of the loss with respect to the linear term \( z \):

$$
\frac{\partial L}{\partial z} = a - y
$$

where:

- \( a = \sigma(z) = \hat{y} \)
- \( y \) is the true label.

### Forward Pass (Vectorized)

We use the **columns-as-examples** convention.

### Shapes

- $ X \in \mathbb{R}^{n \times m} $:

  - $n$ = number of features
  - $m$ = number of training examples
  - Each column $x^{(i)}$ is one example.
- $ w \in \mathbb{R}^{n \times 1} $
- $ b \in \mathbb{R} $ (scalar bias)
- $ Z, A \in \mathbb{R}^{1 \times m} $

### Forward Computation

For each example ( $ i = 1, \dots, m $ ):

$$
z^{(i)} = w^\top x^{(i)} + b
$$
$$
a^{(i)} = \sigma(z^{(i)}) = \frac{1}{1 + e^{-z^{(i)}}}
$$

Stacked (vectorized) form:
$$
Z = w^\top X + b
$$
$$
A = \sigma(Z)
$$

Where the sigmoid function is applied element-wise:
$$
\sigma(Z) = \frac{1}{1 + e^{-Z}}
$$

### NumPy Implementation

```python
z = w.T @ X + b        # shape: (1, m)
A = 1 / (1 + np.exp(-z))
```

Broadcasting automatically expands the scalar (b) to match ((1, m)).
