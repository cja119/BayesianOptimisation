# Bayesian Optimisation of a Gaussian process
Here is a small API which demonstrates the power of Baeysian Optimistaion using a Gauss process, on the following equation:
$$
A \cdot \sin(B \cdot X - C) + D \cdot \cos(E \cdot X) - F + G \cdot X^2 - H \cdot (\exp(-X) - \exp(X))
$$
This is a challenging equation to optimise due to it being multimodal, having lots of local optima.
![Animation](animation.gif)

# Quick Start
To run this API, first clone this repository to your local machine:
```
git clone https://github.com/cja119/BayesianOptimisation.git
```
Then install the necessary dependencies:
```
pip install -r dependencies.txt
```
Finally, run the API:
```
streamlit run bayesian_optimisation.py
```
# Theory

Bayesian Optimization is a method to find the **maximum** (or minimum) of a function, especially when evaluating the function is expensive. It works by creating a **probabilistic model** of the objective function and using that model to decide where to evaluate next.

## Steps in Bayesian Optimization

1. **Model the objective function** using a probabilistic model, usually a Gaussian Process (GP).
2. **Find the next point** to evaluate by optimizing an **acquisition function** that balances exploration (looking in new areas) and exploitation (looking in known good areas).
3. **Update the model** with the new data point and repeat.

## Gaussian Process (GP)

In Bayesian Optimization, we often use a Gaussian Process to model the objective function. A GP assumes that the objective function values follow a multivariate normal distribution.

For any points \( x_1, x_2, \dots, x_n \), the function values \( f(x_1), f(x_2), \dots, f(x_n) \) are jointly normally distributed:

\[
f(x) \sim \mathcal{N}(\mu(x), K(x, x'))
\]

- \( \mu(x) \): Mean function (often assumed to be zero).
- \( K(x, x') \): Covariance function or kernel (describes the similarity between points).

### Kernel Function

The kernel function \( K(x, x') \) defines the covariance between two points. A common kernel is the **squared exponential kernel** (RBF kernel):

\[
K(x, x') = \sigma^2 \exp\left(- \frac{(x - x')^2}{2l^2} \right)
\]

- \( \sigma^2 \): Signal variance.
- \( l \): Length scale (controls how quickly the function varies).

## Acquisition Function

The acquisition function tells us where to sample next. One popular acquisition function is **Expected Improvement (EI)**. It aims to maximize the improvement over the best current value.

The EI at a point \( x \) is:

\[
EI(x) = \mathbb{E}\left[ \max(0, f(x) - f(x_{best})) \right]
\]

Where:
- \( f(x_{best}) \) is the best function value so far.
- \( f(x) \) is the predicted value at point \( x \).

## Bayesian Optimization in Linear Algebra

Given a set of points \( X = [x_1, x_2, \dots, x_n] \) and their corresponding function values \( y = [f(x_1), f(x_2), \dots, f(x_n)] \), we can use Gaussian Process regression to predict the function at a new point \( x_* \).

1. **Compute the Covariance Matrix**: 
   The covariance matrix \( K \) between the points in \( X \) is computed as:
   \[
   K(X, X) = 
   \begin{pmatrix}
   K(x_1, x_1) & \dots & K(x_1, x_n) \\
   \vdots & \ddots & \vdots \\
   K(x_n, x_1) & \dots & K(x_n, x_n)
   \end{pmatrix}
   \]

2. **Compute the Covariance with New Point**:
   The covariance between the new point \( x_* \) and the existing points \( X \) is:
   \[
   k(X, x_*) = 
   \begin{pmatrix}
   K(x_1, x_*) \\
   \vdots \\
   K(x_n, x_*)
   \end{pmatrix}
   \]

3. **Predict the Mean and Variance**:
   The predicted mean \( \mu(x_*) \) and variance \( \sigma^2(x_*) \) at the new point \( x_* \) are given by:

   \[
   \mu(x_*) = k(X, x_*)^T K(X, X)^{-1} y
   \]
   \[
   \sigma^2(x_*) = K(x_*, x_*) - k(X, x_*)^T K(X, X)^{-1} k(X, x_*)
   \]

This gives us both the expected value and uncertainty at the new point \( x_* \), allowing us to update our model.
