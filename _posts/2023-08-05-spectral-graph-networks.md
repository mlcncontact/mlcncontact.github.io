---
layout: post
title:  "Spectral graph networks: generalizing convolutional networks to graph data"
date:   2023-08-05 20:00:00 -0700
---

The inputs to convolutional networks are numbers that lie on regular grids: e.g., a 1D grid for a time-series, or a 2D grid for a grayscale image. For data that lie on graphs that are not regular grids, convolutional networks can't be used (explained in more details [below](#difficulties-of-adapting-convolutions-to-graph-data)). [Bruna et al. (2014)](https://arxiv.org/abs/1312.6203) proposed a way to generalize convolutional networks to graph data, by doing convolutions in the spectral domain. While better-performing graph neural networks have been developed since, I'd like to write about this paper because it contained some very interesting ideas that helped me better understand graph modeling. In this post, I'll first review convolutions and the difficulties of adapting them to graph data, and then explain the ideas behind the spectral network of Bruna et al.

## A brief review of convolutions

The main building block of convolutional networks is the convolutional layer. The basic convolutional layer simply convolves its input with a filter and then applies an elementwise nonlinearity; the filter weights are the parameters being learned. For an input vector $$\mathbf{x}$$, convolution with a filter can be represented as $$\mathbf{C} \mathbf{x}$$, where $$\mathbf{C}$$ is a circulant matrix whose rows are circularly shifted versions of the filter{% include sidenote.html id="circular-convolution" note="Technically, multiplication with a circulant matrix corresponds to a circular convolution, whereas convolutional layers usually use linear convolutions; however, linear convolutions can be turned into circular convolutions by zero-padding." %}.

All circulant matrices share the same set of eigenvectors, which are the Fourier basis vectors. Thus, the convolution can also be written as:

$$
\begin{equation}
\tag{1}
\mathbf{Cx} = \mathbf{F} \mathbf{W} \mathbf{F}^* \mathbf{x}
\end{equation}
$$

where $$\mathbf{F}$$ is the matrix whose columns are the Fourier basis vectors (i.e. $$F_{jk} = \frac{1}{\sqrt{n}} e^{i2\pi (k-1) \frac{j-1}{n}}$$ where $$n$$ is the number of rows/columns of $$\mathbf{F}$$), $$\mathbf{W}$$ is a diagonal matrix containing the Fourier coefficients of the filter, and $$^*$$ denotes conjugate transpose. In words, this is decomposing $$\mathbf{x}$$ into a sum of sines and cosines of different frequencies ($$\mathbf{F}^* \mathbf{x}$$), scaling and shifting each sine and cosine independently from the others ($$\mathbf{W} \mathbf{F}^* \mathbf{x}$$), and then summing up the scaled and shifted sines and cosines ($$\mathbf{F} \mathbf{W} \mathbf{F}^* \mathbf{x}$$).

Typically, the parameters being learned in a convolutional layer are the nonzero elements of $$\mathbf{C}$$ that are repeated across its rows. But the above shows that we could instead have the diagonal of $$\mathbf{W}$$ as the learned parameters. As we will see, this is the idea Bruna et al. used to generalize convolutional layers to graph data.

## Difficulties of adapting convolutions to graph data

{% include marginfigure.html id="abcd_graph" url="assets/img/abcd_graph.svg" description="**Figure 1.** A simple graph with four nodes." %} A graph consists of nodes connected by edges. A simple kind of graph data would be a vector each of whose elements is associated with a node of the graph. An example would be $$\mathbf{x} = (x_a, x_b, x_c, x_d)^T = (3, 9, 4, 5)^T $$ for the graph data illustrated in Figure 1. Note that the ordering of the elements within the vector is arbitrary -- the positional relationship between the elements is specified by the graph, not by their position in the vector. Thus, standard convolution, which assumes that the positional relationship is specified by the position in the vector, wouldn't make sense when applied to this graph data vector. Furthermore, because different nodes can have different numbers of neighbors, it also doesn't make sense to share a single filter across nodes.

Another way to think about these difficulties is to think of our typical, non-graph data as lying on special kinds of graphs. For example, a time-series can be thought of as data lying on a line graph, where sequential time points are sequentially connected nodes. When we represent a time-series as a vector, the position of an element in the vector *is* its position in the graph. Furthermore, nodes in a line graph have the same number of neighbors{% include sidenote.html id="line-graph-ends" note="Padding is needed for this to be true for the nodes at the two ends." %}. Thus, for graphs that are regular grids, convolutions don't run into the difficulties mentioned above.

To get around these difficulties for general graphs, Bruna et al. did convolutions in the spectral domain. But what would be the basis vectors of the spectral domain for a graph? As we'll see below, the eigenvectors of the graph Laplacian can be used as the basis.

## The graph Laplacian and its eigenvectors

A graph Laplacian $$\mathbf{L}$$ is a matrix that reflects the structure of a graph (but not the values of data on the graph). It is a symmetric matrix whose rows correspond to the nodes of the graph and whose columns also correspond to the nodes. Its diagonal element $$L_{ii}$$ is the number of connections to node *i*. Its off-diagonal element $$L_{ij}$$ is -1 if nodes *i* and *j* are connected, and 0 otherwise. For example, for the graph illustrated in Figure 1, its graph Laplacian is

$$
\begin{array}{cc}

&

\begin{array}{cccc}
\hskip0.1em a & \hskip0.8em b & \hskip0.8em c & \hskip0.8em d
\end{array}
\\

\mathbf{L} = 
\begin{array}{c}
a \\
b \\
c \\
d
\end{array}

&

\left(
\begin{array}{cccc}
 1 & -1 &    &    \\
-1 &  3 & -1 & -1 \\
   & -1 &  1 &    \\
   & -1 &    &  1
\end{array}
\right)

\end{array}
$$

One way to understand the eigenvectors of $$\mathbf{L}$$ is to consider the quadratic form $$\mathbf{x}^T \mathbf{L} \mathbf{x}$$ for some data $$\mathbf{x}$$ on the graph. As an illustration, we'll just write out all the terms of $$\mathbf{x}^T \mathbf{L} \mathbf{x}$$ for the graph and data $$\mathbf{x} = (x_a, x_b, x_c, x_d)^T$$ from Figure 1:

$$
\begin{equation}
\begin{split}

\mathbf{x}^T \mathbf{L} \mathbf{x} & = \underbrace{x_a^2 + 3x_b^2 + x_c^2 + x_d^2}_{\substack{\textrm{one square term per} \\ \text{connection to a node}}} \underbrace{- 2x_a x_b - 2x_b x_c - 2x_b x_d}_\textrm{two cross terms for each connection} \\

& = (x_a - x_b)^2 + (x_b - x_c)^2 + (x_b - x_d)^2

\end{split}
\end{equation}

$$

Thus, $$\mathbf{x}^T \mathbf{L} \mathbf{x}$$ is taking the squared difference between each pair of elements of $$\mathbf{x}$$ that are directly connected in the graph, then summing across all the pairs. A low $$\mathbf{x}^T \mathbf{L} \mathbf{x}$$ then means $$\mathbf{x}$$ varies smoothly across the graph, whereas a high $$\mathbf{x}^T \mathbf{L} \mathbf{x}$$ means $$\mathbf{x}$$ varies quickly across the graph.

Now suppose $$\mathbf{v}^{(i)}$$ is an eigenvector of $$\mathbf{L}$$ normalized to length 1. Its eigenvalue is equal to $$\mathbf{v}^{(i)T} \mathbf{L} \mathbf{v}^{(i)}$$, and so it's a measure of how smoothly $$\mathbf{v}^{(i)}$$ varies across the graph. Because $$\mathbf{L}$$ is symmetric, its eigenvectors are mutually orthogonal. Thus, if we order the eigenvectors of $$\mathbf{L}$$ by their eigenvalues, from smallest to largest, we would have an orthogonal set of vectors ordered by their smoothness on the graph: the first eigenvector is the smoothest (the vector all of whose elements are the same, with eigenvalue 0), and each subsequent eigenvector is the smoothest vector orthogonal to all previous eigenvectors.

## Convolutions for graph data

Similar to how standard convolutions can be expressed in the spectral domain as in equation $$(1)$$, we can define convolutions for graph data $$\mathbf{x}$$ as $$\mathbf{V} \mathbf{W} \mathbf{V}^T \mathbf{x}$$, where $$\mathbf{V}$$ is the matrix whose columns are the eigenvectors of the graph Laplacian, and $$\mathbf{W}$$ is a diagonal matrix containing filter weights in the spectral domain. Again, in words, this is decomposing $$\mathbf{x}$$ into a sum of the eigenvectors of the graph Laplacian ($$\mathbf{V}^T \mathbf{x}$$), scaling each eigenvector independently from the others ($$\mathbf{W} \mathbf{V}^T \mathbf{x}$$), and then summing up the scaled eigenvectors ($$\mathbf{V} \mathbf{W} \mathbf{V}^T \mathbf{x}$$).

Graph convolutions defined in this way is not just an operation analogous to standard convolutions; it can be considered a generalization of standard convolutions. Consider the ring graph shown in Figure 2. {% include marginfigure.html id="ring_graph" url="assets/img/ring_graph.svg" description="**Figure 2.** A ring graph. Its graph Laplacian is a circulant matrix." %} Its graph Laplacian is

$$
\mathbf{L} = 
\begin{pmatrix}
 2 & -1 &    &    &    & -1 \\
-1 &  2 & -1 &    &    &    \\
   & -1 &  2 & -1 &    &    \\
   &    & -1 &  2 & -1 &    \\
   &    &    & -1 &  2 & -1 \\
-1 &    &    &    & -1 &  2 \\
\end{pmatrix}
$$

We can see that it is a circulant matrix, which means that its eigenvectors are the Fourier basis{% include sidenote.html id="sine-smoothness" note="In terms of our previous smoothness interpretation, the lower the frequency of a sine wave, the smoother it is." %}. Thus, for the ring graph, transformation to the graph Laplacian eigenvector basis is exactly the Fourier transform, and spectral graph convolution is exactly the standard convolution. In this sense, spectral graph convolution for general graphs is a generalization of the standard convolution.

## The spectral network of Bruna et al.

In the spectral network of Bruna et al., a convolutional layer multiplies its input by $$\mathbf{V} \mathbf{W} \mathbf{V}^T$$ and then applies an elementwise nonlinearity, and the diagonal elements of $$\mathbf{W}$$ are the parameters being learned. In addition, they had two ways of reducing the dimensionality of the spectral domain. First, for a given layer, one could use only a subset of the graph Laplacian eigenvectors for the transform to and inverse transform from the spectral domain. Second, instead of fitting all the diagonal elements of $$\mathbf{W}$$, one could use a low-dimensional basis for those elements, and fit the coefficients of that low-dimensional basis instead.

Finally, I'll mention one limitation of this kind of networks. Because different graphs have different graph Laplacians, a network trained on data from one graph does not transfer to data from other graphs. To be fair, standard convolutional networks actually have the same limitation: a convolutional network trained on images also does not transfer to data from graphs that are not regular grids (of course, we don't think of this as a limitation because we don't need it to transfer outside the domain of images).