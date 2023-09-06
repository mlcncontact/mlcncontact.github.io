---
layout: post
title:  "Exact solutions to the learning dynamics of a deep neural network"
date:   2023-09-03 20:00:00 -0700
---

We don't have a general understanding of how deep neural networks learn. As a step towards a general understanding, [Saxe et al. (2014)](https://arxiv.org/abs/1312.6120) worked out exact solutions to the learning dynamics of a class of simple deep learning problems. In this post I'll write about one of the scenarios they analyzed: linear neural networks with a single hidden layer, squared error loss, and certain simple input patterns and initializations of weights. This is as simple a deep learning problem as you can get--however, despite the linearity of the network, learning here exhibits interesting nonlinear dynamics similar to what are seen with nonlinear networks.

The problem is the following. Our training set consists of $$\{ \mathbf{x}^{\mu}, \mathbf{y}^{\mu} \}$$ for examples $$\mu = 1, 2, ... P$$, where the input $$\mathbf{x}^{\mu}$$ is $$N_1 \times 1$$ and the target $$\mathbf{y}^{\mu}$$ is $$N_3 \times 1$$. Let $$\mathbf{C}^{11} = \sum_{\mu} \mathbf{x}^{\mu} \mathbf{x}^{\mu T} $$ be the $$N_1 \times N_1$$ input correlation matrix, and $$\mathbf{C}^{31} = \sum_{\mu} \mathbf{y}^{\mu} \mathbf{x}^{\mu T} $$ be the $$N_3 \times N_1$$ input-output correlation matrix{% include sidenote.html id="correlation-matrix" note="If the $$P \times 1$$ vector whose $$\mu$$th element is $$y^{\mu}_i$$ has zero mean and unit length, and similarly for the vector whose $$\mu$$th element is $$x^{\mu}_j$$, then $$C^{31}_{ij}$$ is an actual correlation coefficient; otherwise we're using the term \"correlation matrix\" more loosely here." %}. We focus on inputs with orthogonal variables, i.e. $$\mathbf{C}^{11} = \mathbf{I}$$ (this would be true for whitened inputs). We want to map the inputs to the targets using a neural network with a single hidden layer, where $$\mathbf{W}^{21}$$ is the $$N_2 \times N_1$$ weight matrix from the input to the hidden layer, and $$\mathbf{W}^{32}$$ is the $$N_3 \times N_2$$ weight matrix from the hidden layer to the output. Our loss function is $$L = \sum_{\mu} {\| \mathbf{y}^{\mu} - \mathbf{W}^{32} \mathbf{W}^{21} \mathbf{x}^{\mu} \|}^2 $$, on which we perform gradient descent to update the weights $$\mathbf{W}^{21}$$ and $$\mathbf{W}^{32}$$. We are interested in their learning dynamics.

The general approach of Saxe et al. (2014) is to approximate the discrete update steps for the weights with a continuous-time differential equation, and analyze its fixed points and solution. To build intuitions for this approach, let's first apply it to a model that is well-understood--linear regression. Linear regression is closely related to linear neural networks, so it'll also be insightful to compare learning dynamics in the two cases.

## Learning dynamics of linear regression

With linear regression, we are trying to map input $$\mathbf{x}^{\mu}$$ to output $$\mathbf{y}^{\mu}$$ through a single $$N_3 \times N_1$$ matrix $$\mathbf{W}$$, and our loss function is $$L_{lr} = \sum_{\mu} {\| \mathbf{y}^{\mu} - \mathbf{W} \mathbf{x}^{\mu} \|}^2 $$. Its partial derivative with respect to $$W_{ij}$$ is

$$
\begin{equation}
\begin{split}

\frac{\partial L_{lr}}{\partial W_{ij}} & = \sum_{\mu} \sum_{k} \frac{\partial}{\partial W_{ij}} (\mathbf{y}^{\mu} - \mathbf{W} \mathbf{x}^{\mu})_k^2 \\
& = \sum_{\mu} \sum_{k} 2(\mathbf{y}^{\mu} - \mathbf{W} \mathbf{x}^{\mu})_k \frac{\partial}{\partial W_{ij}} (- \sum_{m} W_{km} x_m^{\mu}) \\
& = -2 \sum_{\mu} (\mathbf{y}^{\mu} - \mathbf{W} \mathbf{x}^{\mu})_i x_j^{\mu}

\end{split}
\end{equation}
$$

In matrix form, this is

$$
\begin{equation}
\begin{split}

\nabla_{\mathbf{W}} L_{lr} & = -2 \sum_{\mu} (\mathbf{y}^{\mu} - \mathbf{W} \mathbf{x}^{\mu}) \mathbf{x}^{\mu T} \\
& = -2 (\mathbf{C}^{31} - \mathbf{W} \mathbf{C}^{11})

\end{split}
\end{equation}
$$

Normally we would just set this to 0 and solve for the optimal $$\mathbf{W}$$, but here we are interested in how $$\mathbf{W}$$ changes when it's updated with gradient descent. In this case, the change in $$\mathbf{W}$$ with one gradient descent step is

$$
\begin{equation}

\Delta \mathbf{W} = \lambda (\mathbf{C}^{31} - \mathbf{W} \mathbf{C}^{11})

\end{equation}
$$

where $$\lambda$$ is a small learning rate (the $$2$$ in the gradient has been absorbed into this). We can approximate the discrete update steps with a continuous-time differential equation:

$$
\begin{equation}

\tau \frac{d \mathbf{W}}{dt} = \mathbf{C}^{31} - \mathbf{W} \mathbf{C}^{11}

\end{equation}
$$

Here time $$t$$ is in units of training steps, and $$\tau = 1 / \lambda$$ is a time constant. Because $$\mathbf{C}^{11} = \mathbf{I}$$ for our training set, individual elements of $$\mathbf{W}$$ evolve independently from each other:

$$
\begin{equation}

\tau \frac{d W_{ij}}{dt} = C_{ij}^{31} - W_{ij}

\end{equation}
$$

The fixed point of this equation is $$W_{ij}^{FP} = C_{ij}^{31}$$, and the solution is

$$
\begin{equation}

W_{ij}(t) = W_{ij}(0) e^{-\frac{t}{\tau}} + C_{ij}^{31} (1 - e^{-\frac{t}{\tau}})

\end{equation}
$$

Thus, during learning, each element of $$\mathbf{W}$$ independently and exponentially decays towards its fixed point, with time constant $$\tau$$.

To summarize, for a linear regression model and orthogonal input variables, (1) different weights do not interact with each other, (2) the learning process has a single fixed point ($$\mathbf{W}^{FP} = \mathbf{C}^{31}$$), which is stable, and (3) all weights learn with the same time scale. Next, we consider a linear neural network with a single hidden layer, which simply replaces the weight matrix $$\mathbf{W}$$ of linear regression with a product of two matrices $$\mathbf{W}^{32} \mathbf{W}^{21}$$ as its input-output mapping. However, as we will see, despite the similarity, it exhibits very different learning dynamics, where all three of our conclusions about linear regression are no longer true.