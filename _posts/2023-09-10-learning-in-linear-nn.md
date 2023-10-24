---
layout: post
title:  "Exact solutions to the learning dynamics of a deep neural network"
date:   2023-09-10 20:00:00 -0700
---

We don't have a general understanding of how deep neural networks learn. As a step towards a general understanding, [Saxe et al. (2014)](https://arxiv.org/abs/1312.6120) worked out exact solutions to the learning dynamics of a class of simple deep learning problems. In this post I'll write about one of the scenarios they analyzed: linear neural networks with a single hidden layer, squared error loss, and certain simple input patterns and initializations of weights. This is as simple a deep learning problem as you can get -- however, despite the linearity of the network, learning here exhibits interesting nonlinear dynamics similar to what are seen with some nonlinear networks.

The problem is the following. Our training set consists of $$\{ \mathbf{x}^{\mu}, \mathbf{y}^{\mu} \}$$ for examples $$\mu = 1, 2, ... P$$, where the input $$\mathbf{x}^{\mu}$$ is $$N_1 \times 1$$ and the target $$\mathbf{y}^{\mu}$$ is $$N_3 \times 1$$. Let $$\mathbf{C}^{11} = \sum_{\mu} \mathbf{x}^{\mu} \mathbf{x}^{\mu T} $$ be the $$N_1 \times N_1$$ input correlation matrix, and $$\mathbf{C}^{31} = \sum_{\mu} \mathbf{y}^{\mu} \mathbf{x}^{\mu T} $$ be the $$N_3 \times N_1$$ input-output correlation matrix{% include sidenote.html id="correlation-matrix" note="If the $$P \times 1$$ vector whose $$\mu$$th element is $$y^{\mu}_i$$ has zero mean and unit length, and similarly for the vector whose $$\mu$$th element is $$x^{\mu}_j$$, then $$C^{31}_{ij}$$ is an actual correlation coefficient; otherwise we're using the term \"correlation\" more loosely here." %}. We focus on inputs with orthogonal variables, i.e. $$\mathbf{C}^{11} = \mathbf{I}$$ (this would be true for whitened inputs). We want to map the inputs to the targets using a neural network with a single hidden layer, where $$\mathbf{W}^{21}$$ is the $$N_2 \times N_1$$ weight matrix from the input to the hidden layer, and $$\mathbf{W}^{32}$$ is the $$N_3 \times N_2$$ weight matrix from the hidden layer to the output. Our loss function is $$L = \sum_{\mu} {\| \mathbf{y}^{\mu} - \mathbf{W}^{32} \mathbf{W}^{21} \mathbf{x}^{\mu} \|}^2 $$, on which we perform gradient descent to update the weights $$\mathbf{W}^{21}$$ and $$\mathbf{W}^{32}$$. We are interested in their learning dynamics.

The general approach of Saxe et al. (2014) is to approximate the discrete update steps for the weights with a continuous-time differential equation, and analyze its fixed points and solution. To build intuitions for this approach, let's first apply it to a model that is well-understood -- linear regression. Linear regression is closely related to linear neural networks, so it'll also be insightful to compare learning dynamics in the two cases.

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

This is a linear differential equation; its fixed point is $$W_{ij}^{FP} = C_{ij}^{31}$$, and its solution is

$$
\begin{equation}

W_{ij}(t) = W_{ij}(0) e^{-\frac{t}{\tau}} + C_{ij}^{31} (1 - e^{-\frac{t}{\tau}})

\end{equation}
$$

Thus, during learning, each element of $$\mathbf{W}$$ independently and exponentially decays towards its fixed point, with time constant $$\tau$$.

To summarize, for a linear regression model and orthogonal input variables, (i) the learning dynamics is linear, (ii) different weights do not interact with each other during learning, (iii) the learning process has a single fixed point ($$\mathbf{W}^{FP} = \mathbf{C}^{31}$$), which is stable, and (iv) all weights learn with the same timescale. Next, we consider a linear neural network with a single hidden layer, which simply replaces the weight matrix $$\mathbf{W}$$ of linear regression with a product of two matrices $$\mathbf{W}^{32} \mathbf{W}^{21}$$ as its input-output mapping. However, as we will see, despite the similarity, it exhibits very different learning dynamics, where all four of our conclusions about linear regression are no longer true.

## Learning dynamics of linear neural networks

As before, we start with our loss function $$L = \sum_{\mu} {\| \mathbf{y}^{\mu} - \mathbf{W}^{32} \mathbf{W}^{21} \mathbf{x}^{\mu} \|}^2 $$, and take its partial derivative with respect to an element of the input-to-hidden weight matrix:

$$
\begin{equation}
\begin{split}

\frac{\partial L}{\partial W_{ij}^{21}} & = \sum_{\mu} \sum_{k} \frac{\partial}{\partial W_{ij}^{21}} (\mathbf{y}^{\mu} - \mathbf{W}^{32} \mathbf{W}^{21} \mathbf{x}^{\mu})_k^2 \\
& = \sum_{\mu} \sum_{k} 2(\mathbf{y}^{\mu} - \mathbf{W}^{32} \mathbf{W}^{21} \mathbf{x}^{\mu})_k \frac{\partial}{\partial W_{ij}^{21}} (- \sum_{m} \sum_{n} W_{kn}^{32} W_{nm}^{21} x_m^{\mu}) \\
& = -2 \sum_{\mu} \sum_{k} (\mathbf{y}^{\mu} - \mathbf{W}^{32} \mathbf{W}^{21} \mathbf{x}^{\mu})_k W_{ki}^{32} x_j^{\mu} \\
& = -2 \sum_{\mu} [\mathbf{W}^{32T} (\mathbf{y}^{\mu} - \mathbf{W}^{32} \mathbf{W}^{21} \mathbf{x}^{\mu})]_i x_j^{\mu}

\end{split}
\end{equation}
$$

In matrix form, this is

$$
\begin{equation}
\begin{split}

\nabla_{\mathbf{W}^{21}} L & = -2 \sum_{\mu} \mathbf{W}^{32T} (\mathbf{y}^{\mu} - \mathbf{W}^{32} \mathbf{W}^{21} \mathbf{x}^{\mu}) \mathbf{x}^{\mu T} \\
& = -2 \mathbf{W}^{32T} (\mathbf{C}^{31} - \mathbf{W}^{32} \mathbf{W}^{21} \mathbf{C}^{11})

\end{split}
\end{equation}
$$

Again, the gradient update steps for $$\mathbf{W}^{21}$$ can be approximated with a differential equation:

$$
\begin{equation}
\tag{1}

\tau \frac{d \mathbf{W}^{21}}{dt} = \mathbf{W}^{32T} (\mathbf{C}^{31} - \mathbf{W}^{32} \mathbf{W}^{21})

\end{equation}
$$

where we've assumed $$\mathbf{C}^{11} = \mathbf{I}$$. We can do similar calculations to obtain the equation for $$\mathbf{W}^{32}$$:

$$
\begin{equation}
\tag{2}

\tau \frac{d \mathbf{W}^{32}}{dt} = (\mathbf{C}^{31} - \mathbf{W}^{32} \mathbf{W}^{21}) \mathbf{W}^{21T}

\end{equation}
$$

These equations show that the weights no longer evolve independently of each other. For example, consider $$W_{ij}^{21}$$:

$$
\begin{equation}

\tau \frac{d W_{ij}^{21}}{dt} = \sum_{k} W^{32}_{ki} C^{31}_{kj} - \sum_{k} \sum_{m} W^{32}_{ki} W^{32}_{km} W^{21}_{mj}

\end{equation}
$$

Thus, changes in one weight depends on up to cubic interactions involving weights in the same layer as well as weights in a different layer. This is not easy to understand; to simplify these interactions, we will change to more convenient bases for the input and output spaces -- bases consisting of the right and left singular vectors of the input-output correlation matrix $$\mathbf{C}^{31}$$. For this, let the singular value decomposition of $$\mathbf{C}^{31}$$ be $$\mathbf{C}^{31} = \mathbf{U}^{33} \mathbf{S}^{31} \mathbf{V}^{11T} $$. The columns of the $$N_3 \times N_3$$ matrix $$\mathbf{U}^{33}$$ are an orthonormal basis for the output space; the diagonal of the $$N_3 \times N_1$$ matrix $$\mathbf{S}^{31}$$ contains the singular values $$s_\alpha$$'s in descending order; the columns of the $$N_1 \times N_1$$ matrix $$\mathbf{V}^{11}$$ are an orthonormal basis for the input space.

We will now consider the transformed weight matrices $$\overline{\mathbf{W}}^{21} = \mathbf{W}^{21} \mathbf{V}^{11}$$ and $$\overline{\mathbf{W}}^{32} = \mathbf{U}^{33T} \mathbf{W}^{32} $$. Here $$\overline{\mathbf{W}}^{21}_{i \alpha}$$ is the connection strength from input mode $$\alpha$$ (input in the direction of the $$\alpha$$th right singular vector) to hidden unit $$i$$; $$\overline{\mathbf{W}}^{32}_{\alpha i}$$ is the connection strength from hidden unit $$i$$ to output mode $$\alpha$$ (output in the direction of the $$\alpha$$th left singular vector). With these transformed variables, equations $$(1)$$ and $$(2)$$ become:

$$
\begin{equation}
\tag{3}

\tau \frac{d \overline{\mathbf{W}}^{21}}{dt} = \overline{\mathbf{W}}^{32T} (\mathbf{S}^{31} - \overline{\mathbf{W}}^{32} \overline{\mathbf{W}}^{21})

\end{equation}
$$

$$
\begin{equation}
\tag{4}

\tau \frac{d \overline{\mathbf{W}}^{32}}{dt} = (\mathbf{S}^{31} - \overline{\mathbf{W}}^{32} \overline{\mathbf{W}}^{21}) \overline{\mathbf{W}}^{21T} 

\end{equation}
$$

In the following, I'll use $$\mathbf{A}_{:i}$$ to denote the $$i$$th column of a matrix $$\mathbf{A}$$, and $$\mathbf{A}_{i:}$$ to denote the $$i$$th row. Let $$\mathbf{a}^{\alpha} = \overline{\mathbf{W}}^{21}_{:\alpha}$$; this is a vector in the space of hidden units -- input mode $$\alpha$$ activates hidden units along this direction. Let $$\mathbf{b}^{\alpha T} = \overline{\mathbf{W}}^{32}_{\alpha :}$$; this is another vector in hidden units space -- hidden unit activity along this direction activates output mode $$\alpha$$. For simplicity, we'll assume that $$N_1 = N_3$$, so that there are equal numbers of input and output modes ($$N_1 \neq N_3$$ adds a bit of uninteresting complexity); this would be the case if the network is used as an autoencoder, for example. Then, from equation $$(3)$$, we can find the learning dynamics of $$\mathbf{a}^{\alpha}$$:

$$
\begin{align}

\tau \frac{d \overline{\mathbf{W}}^{21}_{:\alpha}}{dt} &= \overline{\mathbf{W}}^{32T} (\mathbf{S}^{31} - \overline{\mathbf{W}}^{32} \overline{\mathbf{W}}^{21})_{:\alpha} \\
\tau \frac{d \mathbf{a}^{\alpha}}{dt} &= (\overline{\mathbf{W}}^{32T})_{:\alpha} s_{\alpha} - \overline{\mathbf{W}}^{32T} \overline{\mathbf{W}}^{32} \overline{\mathbf{W}}^{21}_{:\alpha} \\
\tau \frac{d \mathbf{a}^{\alpha}}{dt} &= s_{\alpha} \mathbf{b}^{\alpha} - \sum_{\gamma = 1}^{N_3} (\mathbf{b}^{\gamma T} \mathbf{a}^{\alpha}) \mathbf{b}^{\gamma} \\

\tau \frac{d \mathbf{a}^{\alpha}}{dt} &= (s_{\alpha} - \mathbf{b}^{\alpha T} \mathbf{a}^{\alpha}) \mathbf{b}^{\alpha} - \sum_{\gamma \neq \alpha} (\mathbf{b}^{\gamma T} \mathbf{a}^{\alpha}) \mathbf{b}^{\gamma} \tag{5}

\end{align}
$$

Similarly, from equation $$(4)$$ we can find the learning dynamics of $$\mathbf{b}^{\alpha}$$ to be

$$
\begin{align}

\tau \frac{d \mathbf{b}^{\alpha}}{dt} &= (s_{\alpha} - \mathbf{b}^{\alpha T} \mathbf{a}^{\alpha}) \mathbf{a}^{\alpha} - \sum_{\gamma \neq \alpha} (\mathbf{b}^{\alpha T} \mathbf{a}^{\gamma}) \mathbf{a}^{\gamma} \tag{6}

\end{align}
$$

These equations make it much easier to understand how weights interact during learning. For each $$\alpha$$, the first term of equation $$(5)$$ and equation $$(6)$$ drives $$\mathbf{a}^{\alpha}$$ and $$\mathbf{b}^{\alpha}$$ to point in the same direction, and it drives their dot product towards $$s_{\alpha}$$. The second term of equation $$(5)$$ shrinks the components of $$\mathbf{a}^{\alpha}$$ in the directions of $$\mathbf{b}^{\gamma}$$'s for $$\gamma \neq \alpha$$, whereas the second term of equation $$(6)$$ shrinks the components of $$\mathbf{b}^{\alpha}$$ in the directions of $$\mathbf{a}^{\gamma}$$'s for $$\gamma \neq \alpha$$. 

We can see that equations $$(5)$$ and $$(6)$$ correspond to gradient descent on the loss function $$ \frac{1}{2\tau} \sum_{\alpha} (s_{\alpha} - \mathbf{b}^{\alpha T} \mathbf{a}^{\alpha})^2 + \frac{1}{2\tau} \sum_{\alpha \neq \beta} (\mathbf{b}^{\beta T} \mathbf{a}^{\alpha})^2 $$, which penalizes deviations of $$\mathbf{a}^{\alpha}$$'s and $$\mathbf{b}^{\beta}$$'s from $$\mathbf{b}^{\beta T} \mathbf{a}^{\alpha} = s_{\alpha} \delta_{\alpha \beta}$$ (where $$\delta_{\alpha \beta}$$ is the Kronecker delta). The dot product $$\mathbf{b}^{\beta T} \mathbf{a}^{\alpha}$$ is the extent to which input mode $$\alpha$$ drives output mode $$\beta$$; thus, learning drives the network towards a decoupled regime, where each input mode $$\alpha$$ drives only output mode $$\alpha$$ and none of the other output modes, and it drives output mode $$\alpha$$ to the extent of their input-output correlation $$s_{\alpha}$$ from the training set.

Why does learning drive $$\mathbf{b}^{\alpha T} \mathbf{a}^{\alpha}$$ towards $$s_{\alpha}$$? As we saw above, the optimal input-output mapping we found from linear regression was $$\mathbf{W}^{FP} = \mathbf{C}^{31} = \mathbf{U}^{33} \mathbf{S}^{31} \mathbf{V}^{11T}$$. $$\mathbf{W}^{FP}$$ expresses the optimal input-output mapping in the bases of input variables and output variables, where $$\mathbf{W}^{FP}_{ij}$$ is how much input variable $$j$$ should contribute to output variable $$i$$. The $$s_{\alpha}$$'s express the optimal input-output mapping in the bases of singular vectors, where $$s_{\alpha}$$ is how much input mode $$\alpha$$ should contribute to output mode $$\alpha$$.

Next, let's consider the fixed points of equations $$(5)$$ and $$(6)$$. First, note that $$\mathbf{a}^{\alpha} = \mathbf{b}^{\alpha} = \mathbf{0}$$ for all $$\alpha$$'s is a fixed point. If $$\mathbf{a}^{\alpha}$$ and $$\mathbf{b}^{\alpha}$$ are nonzero for some $$\alpha$$'s, then the condition for a fixed point is for the nonzero $$\mathbf{a}^{\alpha}$$'s and $$\mathbf{b}^{\alpha}$$'s to satisfy $$\mathbf{b}^{\beta T} \mathbf{a}^{\alpha} = s_{\alpha} \delta_{\alpha \beta}$${% include sidenote.html id="fixed-point-family" note="Note that $$\mathbf{b}^{\beta T} \mathbf{a}^{\alpha} = \frac{1}{k} \mathbf{b}^{\beta T} k \mathbf{a}^{\alpha}$$ for some scaling factor $$k$$, so any fixed point is associated with a family of fixed points that differ in the scaling of $$\mathbf{a}^{\alpha}$$'s and $$\mathbf{b}^{\alpha}$$'s." %}. If $$N_2 < N_1$$ and $$N_2 < N_3$$, then there can be at most $$N_2$$ pairs of nonzero $$\mathbf{a}^{\alpha}$$'s and $$\mathbf{b}^{\alpha}$$'s that satisfy this condition (because there cannot be more than $$N_2$$ mutually orthogonal vectors in a $$N_2$$-dimensional space). Suppose the rank of $$\mathbf{C}^{31}$$ is $$R$$ and $$R > N_2$$. We can get a fixed point by picking any $$N_2$$ pairs of $$\mathbf{a}^{\alpha}$$'s and $$\mathbf{b}^{\alpha}$$'s to satisfy $$\mathbf{b}^{\beta T} \mathbf{a}^{\alpha} = s_{\alpha} \delta_{\alpha \beta}$$, and have $$\mathbf{a}^{\alpha} = \mathbf{b}^{\alpha} = \mathbf{0}$$ for the remaining $$R - N_2$$ pairs. [Baldi and Hornik (1989)](https://www.sciencedirect.com/science/article/abs/pii/0893608089900142) showed that the global minimum is when the nonzero $$\mathbf{a}^{\alpha}$$'s and $$\mathbf{b}^{\alpha}$$'s correspond to the $$N_2$$ largest $$s_{\alpha}$$'s, and that all other fixed points are saddle points. At the global minimum, the input-output mapping of the network is

$$
\begin{equation}
\begin{split}

\mathbf{W}^{32} \mathbf{W}^{21} &= \mathbf{U}^{33} \mathbf{U}^{33T} \mathbf{W}^{32} \mathbf{W}^{21} \mathbf{V}^{11} \mathbf{V}^{11T} \\
&= \mathbf{U}^{33} \overline{\mathbf{W}}^{32} \overline{\mathbf{W}}^{21} \mathbf{V}^{11T} \\
&= \sum_{\alpha = 1}^{N_2} s_{\alpha} \mathbf{U}^{33}_{:\alpha} \mathbf{V}^{11T}_{:\alpha}

\end{split}
\end{equation}
$$

which is the best rank-$$N_2$$ approximation of $$\mathbf{W}^{FP} = \mathbf{C}^{31}$$ (Eckart-Young-Mirsky theorem).

Next, we will solve equations $$(5)$$ and $$(6)$$ for a special class of weight initializations, where the network is in a decoupled regime to start with, to better understand the timescales of learning. Let $$\mathbf{r}^{\alpha}$$ for $$\alpha = 1, 2, ... N_2$$ be a set of $$N_2 \times 1$$ orthonormal vectors. We initialize the weights at $$\mathbf{a}^{\alpha}(0) = \mathbf{b}^{\alpha}(0) \propto \mathbf{r}^{\alpha}$$ for $$\alpha = 1, 2, ... N_2$$ and $$\mathbf{a}^{\alpha}(0) = \mathbf{b}^{\alpha}(0) = \mathbf{0}$$ for $$\alpha > N_2$$. Because of the orthogonality of the $$\mathbf{r}^{\alpha}$$'s, the second term of equation $$(5)$$ and equation $$(6)$$ are $$\mathbf{0}$$. Because $$\mathbf{a}^{\alpha}(0)$$ and $$\mathbf{b}^{\alpha}(0)$$ point in the same direction, the first term of equation $$(5)$$ and equation $$(6)$$ will only change the magnitudes of $$\mathbf{a}^{\alpha}$$ and $$\mathbf{b}^{\alpha}$$, not their directions. Thus, we can write $$\mathbf{a}^{\alpha}(t) = a_{\alpha}(t) \mathbf{r}^{\alpha}$$ and $$\mathbf{b}^{\alpha}(t) = b_{\alpha}(t) \mathbf{r}^{\alpha}$$. Plugging these into equations $$(5)$$ and $$(6)$$, we find

$$
\begin{align}

\tau \frac{d a_{\alpha}}{dt} &= (s_{\alpha} - a_{\alpha} b_{\alpha}) b_{\alpha} \tag{7} \\
\tau \frac{d b_{\alpha}}{dt} &= (s_{\alpha} - a_{\alpha} b_{\alpha}) a_{\alpha} \tag{8}

\end{align}
$$

The global minimum fixed point in terms of the magnitudes is now $$a_{\alpha} b_{\alpha} = s_{\alpha}$$ for $$\alpha = 1, 2, ... N_2$$ and $$a_{\alpha} = b_{\alpha} = 0$$ for $$\alpha > N_2$$. Since we've assumed that $$a_{\alpha}(0) = b_{\alpha}(0)$$, we know that $$a_{\alpha}(t) = b_{\alpha}(t)$$ for all $$t$$, and we can simply consider the evolution of $$u_{\alpha} = a_{\alpha} b_{\alpha}$$:

$$
\begin{equation}

\tau \frac{d u_{\alpha}}{dt} = 2a_{\alpha} \tau \frac{d a_{\alpha}}{dt} = 2 u_{\alpha} (s_{\alpha} - u_{\alpha})

\end{equation}
$$

This equation can be solved via separation of variables:

$$
\begin{align}

\frac{\tau}{2 u_{\alpha} (s_{\alpha} - u_{\alpha})} \frac{d u_{\alpha}}{dt} &= 1 \\
\frac{\tau}{2} \int_{u_{\alpha}(0)}^{u_{\alpha}(t)} \frac{1}{u_{\alpha} (s_{\alpha} - u_{\alpha})} du_{\alpha} &= \int_{0}^{t} 1 dt' \\
\frac{\tau}{2s} \ln \frac{u_{\alpha}(t) [s_{\alpha} - u_{\alpha}(0)]}{u_{\alpha}(0) [s_{\alpha} - u_{\alpha}(t)]} &= t

\end{align}
$$

Rearranging this, we find the solution to be

$$
\begin{equation}
\tag{9}

u_{\alpha}(t) = \frac{s_{\alpha} e^{2 s_{\alpha} t / \tau}}{e^{2 s_{\alpha} t / \tau} - 1 + s_{\alpha} / u_{\alpha}(0)}

\end{equation}
$$

Now let's think about what this means. {% include marginfigure.html id="learning_curves" url="assets\img\linear_mode_learning_curves.svg" description="**Figure 1.** Taken from Saxe et al. (2014). The curves are input-output mapping strengths for a number of $$\alpha$$'s; each color is a different network.  Red curves show equation $$(9)$$ (linear network initialized in the decoupled regime). Blue curves show a simulation of a linear network with random weight initializations. Green curves show a simulation of a nonlinear network with $$\tanh$$ activation functions." %}$$u_{\alpha} = a_{\alpha} b_{\alpha} \mathbf{r}^{\alpha T} \mathbf{r}^{\alpha} = \mathbf{b}^{\alpha T} \mathbf{a}^{\alpha}$$ is the extent to which input mode $$\alpha$$ drives output mode $$\alpha$$, with $$\mathbf{r}^{\alpha}$$ being a direction in hidden unit space that serves as a relay between the two modes. Each $$u_{\alpha}$$, and thus the connection strength between each pair of input mode $$\alpha$$ and output mode $$\alpha$$, learns independently. $$u_{\alpha}$$ is a sigmoid function of time, approaching the optimal connection strength $$s_{\alpha}$$ as $$t \to \infty$$, with effective time constant $$\frac{\tau}{2 s_{\alpha}}$$. $$s_{\alpha}$$ is the correlation strength between input mode $$\alpha$$ and output mode $$\alpha$$ in the training set; thus, the stronger the correlation, the faster the network learns to map an input mode to its corresponding output mode. Figure 1, taken from Saxe et al. (2014), shows equation $$(9)$$ for a number of $$\alpha$$'s, where we can see their different timescales of learning. It also shows learning dynamics for the same linear network but with random weight initializations, as well as a nonlinear version of the network; we can see that equation $$(9)$$ provides a reasonable approximation to both.

As a final note, let's consider a phenomenon that is sometimes observed when training nonlinear deep neural networks. During training, sometimes the loss function plateaus for a while, and then drops sharply to a lower level. It's possible that the learning dynamics of equation $$(9)$$ and Figure 1 offers an explanation: the plateau could be when learning for one $$u_{\alpha}$$ has saturated while $$u_{\alpha + 1}$$ has not reached the rapidly rising phase of its sigmoid, and when it does reach it the loss drops quickly.

## References

Baldi, P., & Hornik, K. (1989). Neural networks and principal component analysis: Learning from examples without local minima. *Neural Networks*.

Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. *ICLR*.
