---
title: "Poster: FOM Veldhoven 2016"
description: Poster presented at FOM Veldhoven 2016
date: 2016-01-18
layout: pdf
categories: posters
tags: gaussian-processes,machine-learning,space-weather
file: FOM_Veldhoven_2016_poster.pdf
permalink: /posters/fom-veldhoven-2016/
---

## Gaussian Process Regression Models for Space Weather Prediction.

### Space Weather

![Aurora]({{ site.url }}/public/aurora.jpg)

>Space weather is a branch of space physics and aeronomy concerned with the time varying conditions within the Solar System,
>including the solar wind, emphasizing the space surrounding the Earth, including conditions in the magnetosphere, ionosphere
>and thermosphere.
>
>Space weather is distinct from the terrestrial weather of the Earth's atmosphere (troposphere and stratosphere).  
>
>Source: [Wikipedia](https://en.wikipedia.org/wiki/Space_weather)


### Geomagnetic Activity Indices

Space Weather exhibits complex non-linear dynamics due to the high number of variables that one encounters in the physics based models, their inter dependences and complexity of the governing equations. It is therefore instructive from the point of view of prediction, to condense the geomagnetic response of the Earth to a set of representative indices, some of which are summarized in the table below.


Name | Significance | Frequency | Values
------------ | ------------- | -------------|-------------
Kp   | Global geomagnetic storm index and is based on 3 hour measurements of the K-indices, for a given value, for each of the past days | 3 hours | 0-9  
Dst | Average ring current around magnetic equator | hourly | Real Number  
AE | The AE index is derived from geomagnetic variations in the horizontal component observed at selected (10-13) observatories along the auroral zone in the northern hemisphere | hourly | Real Number

####D<sub>st</sub>

We focus on modeling the [Disturbance Storm Time](https://en.wikipedia.org/wiki/Disturbance_storm_time_index), though our results can be generalised for the AE and K<sub>p</sub> indices as well. The chart below explains how different values of D<sub>st</sub> relate to the state of the Earth's magnetosphere.

![Dst as a time series]({{ site.url }}/public/dst.png)

Thus the modeling of D<sub>st</sub> is important in the recognition and prediction of geo-magnetic disturbances and storms.



###Gaussian Process Regression

Given below is the formulation of a *Gaussian Process* regression model. For a detailed introduction on *Gaussian Processes* you can refer to the book written by [Ramussen and Williams](https://books.google.nl/books/about/Gaussian_Processes_for_Machine_Learning.html?id=vWtwQgAACAAJ&hl=en).

####Assumptions

We assume that our target data are noisy observations of an unknown function $$ f(x) $$. This modeling assumption leads to a _Stochastic Process_ formulation for the prior distribution on this unknown function.

The existence of such a _stochastic process_ is established in the [Kolmogorov Extension Theorem](https://en.wikipedia.org/wiki/Kolmogorov_extension_theorem) with the assumption of existence of a positive semi-definite, symmetric covariance function $$ C(x,y): \Omega \times \Omega \rightarrow \mathbb{R} \ \ x,y \in \Omega$$.

We further assume that the finite dimensional distributions are multivariate gaussian, leading to the following set of equations for the finite dimensional distributions of the unknown function $$f(x)$$.


####Formulation

$$
	\begin{align}
		& y = f(x) + \epsilon \\
		& f \sim \mathcal{GP}(m(x), C(x,x')) \\
		& \left(\mathbf{y} \ \ \mathbf{f_*} \right)^T \sim \mathcal{N}\left(\mathbf{0}, \left[ \begin{matrix} K(X, X) + \sigma^{2} \it{I} & K(X, X_*) \\ K(X_*, X) & K(X_*, X_*) \end{matrix} \right ] \right) 

	\end{align}
$$
  
####Posterior Predictive Distribution
In the presence of training data $$ X = (x_1, x_2, \cdot , x_n) \ y = (y_1, y_2, \cdot , y_n) $$, one may calculate using _Bayes Theorem_ the posterior predictive distribution $$ \mathbf{f_*}|X,\mathbf{y},X_* $$ assuming $$ X_* $$, the test inputs are known.


$$
	\begin{align}
		& \mathbf{f_*}|X,\mathbf{y},X_* \sim \mathcal{N}(\mathbf{\bar{f_*}}, cov(\mathbf{f_*}))  \label{eq:posterior}\\
		& \mathbf{\bar{f_*}} \overset{\triangle}{=} \mathbb{E}[\mathbf{f_*}|X,y,X_*] = K(X_*,X)[K(X,X) + \sigma^{2}_n \it{I}]^{-1} \mathbf{y} \label{eq:posterior:mean} \\
		& cov(\mathbf{f_*}) = K(X_*,X_*) - K(X_*,X)[K(X,X) + \sigma^{2}_n \it{I}]^{-1}K(X,X_*) 
	
	\end{align}
$$


###Gaussian Process D<sub>st</sub> models: RBF vs FBM Kernels.

We model D<sub>st</sub> as a scalar valued function of the solar wind speed. 

$$
	\begin{align}
		& Dst(v) \sim \mathcal{GP}(m(v), C(u,v)) \label{eq:DstGP}\\
		& C_{rbf}(u,v) = \mathbb{E}[Dst(u) \times Dst(v)] =  e^{-\frac{1}{2}|u-v|^2/\sigma^2} \label{eq:rbfcov}\\
		& C_{fbm}(u,v) = \mathbb{E}[Dst(u) \times Dst(v)] = |u|^{2H} + |v|^{2H} - |u-v|^{2H} \label{eq:fbmcov}
	\end{align}
$$
  
In the equations above, $$ H \in (0,1] $$ and $$ \sigma $$ are the hyper-parameters of the _Fractional Brownian_ and _Radial Basis Function_ kernels respectively, when training _Gaussian Process_ models, it is imperetive to choose optimal values of these hyper-parameters which can be achieved by a number of means (refer [Ramussen and Williams](https://books.google.nl/books/about/Gaussian_Processes_for_Machine_Learning.html?id=vWtwQgAACAAJ&hl=en) for further details). Some of the common algorithms applied for this purpose are based on maximum likelihood and cross-validation techniques. In this case we have used maximum likelihood driven search on a pre-defined grid of hyper-parameters.

We compare the performance of two _Gaussian Process_ regression models for D<sub>st</sub>, one with the _Radial Basis Function_ kernel given by $$ C_{rbf} $$ and the _Fractional Brownian Motion_ kernel given by $$ C_{fbm} $$. Both models are trained and tested on sub-sampled versions of the Omni data from the years 2007 and 2006 respectively.

###Results

The performance metrics of both the constructed models on the test set are summarized below.

Kernel | Data: Train, Test| MAE | RMSE | R<sup>2</sup> 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
RBF|300, 1000 | 1.5044 | 6.9752  | 0.7925  
FBM|300, 1000 | 0.0312  | 0.0461  | 0.9999

While both models do a decent job of predicting the D<sub>st</sub> index given the solar wind velocity, a closer look at the residual histograms and goodness of fit charts shows the differences between them.


#### Radial Basis Function Kernel

The _Radial Basis Function_ kernel which tries to fit smooth splines to the data exhibits an interesting pathology: it is unable to predict anamalous geo-magnetic conditions, namely it can not predict with sufficient reliability the onset of geo-magnetic storms ($$ D_{st} \leq -100 nT $$). This can be observed in the long tails in its error distribution and on the goodness of fit one can clearly observe those points as being far away from the "best fit" regression line.

Plot| RBF Kernel 
:-------------------------:|:-------------------------:
Fit|<img src="{{ site.url }}/public/rbf-fit.png" width="600">  |   
Histogram|<img src="{{ site.url }}/public/rbf-hist.png" width="600">  |  




#### Fractional Brownian Motion Kernel

The _Fractional Brownian_ kernel gives far more accurate D<sub>st</sub> predictions for both slow and turbulent solar wind conditions, pointing to the idea that fitting smooth splines is not a reasonable modeling assumption when learning models of the form $$ D_{st}(v_{solar wind})$$. 

Plot | FBM Kernel 
:-------------------------:|:-------------------------:
Fit|<img src="{{ site.url }}/public/fbm-fit.png" width="600">  |   
Histogram|<img src="{{ site.url }}/public/fbm-hist.png" width="600">  |  
