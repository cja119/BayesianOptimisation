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

1. **Model the objective function** using a probabilistic model.
2. **Find the next point** to evaluate by optimizing an **acquisition function** that balances exploration (looking in new areas) and exploitation (looking in known good areas).
3. **Update the model** with the new data point and repeat.

## Gaussian Process (GP)

As is often the case with Bayesian Optimsiation, we are using a Gauss Process as our probablistic model. 

For any points \( $x_1, x_2, \dots, x_n $\), the function values \( $f(x_1), f(x_2), \dots, f(x_n) $\) are jointly normally distributed:

$$
f(x) \sim \mathcal{N}(\mu(x), K(x, x'))
$$

- \( $\mu(x)$ \): Mean function (often assumed to be zero).
- \( $K(x, x')$ \): Covariance function or kernel (describes the similarity between points).
- \( $K(x, x')$ \): Covariance function or kernel (describes the similarity between points).

### Kernel Function

The kernel function \( K(x, x') \) defines the covariance between two points. Here we will use the **squared exponential kernel** (RBF kernel):

$$
K(x, x') = \sigma^2 \exp\left(- \frac{(x - x')^2}{2l^2} \right)
$$

- \( $\sigma^2 $\): Signal variance.
- \( $l $\): Length scale (controls how quickly the function varies).

## Acquisition Function

The acquisition function tells us where to sample next. For our model, we use our uncertainty to help guide our next choice.

$$
x = \text{argmin}\[(\mu (x_\text{pred}) - n \cdot \sigma(x_\text{pred}))\]
$$

Where:
- \( $\mu (x_\text{pred}) )$ \) is a vector of the Gauss Process' average predictions (accross our X domain)
- \( $n$ \) is a hyperparameter.
- \( $\sigma (x_\text{pred}) )$ \) is a vector of the `uncertainty' in our average predicitons (ie, the variance)

In order to also ensure our model explores the space (useful if there are local optima) it has an improvement threshhold, below wich it will use a different acquisition function:

$$
x = \text{argmin}\[\sigma(x_\text{pred}))\]
$$

This chooses the next x value based on that which will provide the greatest reduction in uncertainty in our model.

# Use of Generative Ai

Please note that generative AI has been used in places to generate the code found in this repository. If there are any copyright issues resulting from this, please reach out to me directly.
