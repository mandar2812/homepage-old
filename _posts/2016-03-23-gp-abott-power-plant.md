---
title: "System Identification using Gaussian Processes: Abott Power Plant, Champaign, Illinois"
description: Tutorial about Gaussian Process NARX models on Abott power plant data using DynaML
date: 2016-03-23
layout: post
categories: posts
tags: gaussian-process machine-learning system-identification kernel-methods DynaML
comments: True
permalink: /posts/gp-abott-power-plant/
---

------

In this post, we use the [DynaML](https://transcendent-ai-labs.github.io/DynaML/) Scala machine learning environment to train Gaussian Process models to analyse time series data taken from a coal power plant.

<br/>

![Abott: Representative Image]({{ site.url }}/public/abott.jpg)

<br/>

### The Data Set

From the [Daisy](http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html) system identification database, we download the [abott power plant data](ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/steamgen.dat.gz). The data characteristics are summarized below.

**Description**:
	The data comes from a model of a Steam Generator at
	Abbott Power Plant in Champaign IL.

**Sampling Frequency**:
	3 sec

**Number**:
	9600

**Inputs**:
	1. Fuel scaled 0-1
	2. Air	scaled 0-1
	3. Reference level inches
	4. Disturbance definde by the load level
	
**Outputs**:
	5. Drum pressure PSI
	6. Excess Oxygen in exhaust gases %
	7. Level of water in the drum
	8. Steam Flow Kg./s

### Nonlinear AutoRegressive with eXogenous inputs (NARX)
A candidate output signal $$y(t)$$ modeled as a function of the previous $$p$$ values of itself and the $$m$$ exogenous inputs $$u_{1}, \cdots u_{m}$$

$$
	\begin{align}
    \begin{split}
        y(t) = & f(y(t-1), y(t-2), \cdots, y(t-p), \\ 
        & u_{1}(t-1), u_{1}(t-2), \cdots, u_{1}(t-p),\\
        & \cdots, \\
        & u_{m}(t-1), u_{m}(t-2), \cdots, u_{m}(t-p)) \\
        & + \epsilon(t)
    \end{split}
	\end{align}
$$

<br/>

### Gaussian Processes

Gaussian Processes are powerful non-parametric methods to solve regression and classification problems. They are based on a structural assumption about the finite dimensional distributions over spaces of functions, as shown in the equations below.

#### Formulation

$$
	\begin{align}
		& y = f(x) + \epsilon \\
		& f \sim \mathcal{GP}(m(x), C(x,x')) \\
		& \left(\mathbf{y} \ \ \mathbf{f_*} \right)^T \sim \mathcal{N}\left(\mathbf{0}, \left[ \begin{matrix} K(X, X) + \sigma^{2} \it{I} & K(X, X_*) \\ K(X_*, X) & K(X_*, X_*) \end{matrix} \right ] \right) 

	\end{align}
$$
  
#### Posterior Predictive Distribution
In the presence of training data $$ X = (x_1, x_2, \cdot , x_n) \ y = (y_1, y_2, \cdot , y_n) $$, one may calculate using _Bayes Theorem_ the posterior predictive distribution $$ \mathbf{f_*}|X,\mathbf{y},X_* $$ assuming $$ X_* $$, the test inputs are known.


$$
	\begin{align}
		& \mathbf{f_*}|X,\mathbf{y},X_* \sim \mathcal{N}(\mathbf{\bar{f_*}}, cov(\mathbf{f_*}))  \label{eq:posterior}\\
		& \mathbf{\bar{f_*}} \overset{\triangle}{=} \mathbb{E}[\mathbf{f_*}|X,y,X_*] = K(X_*,X)[K(X,X) + \sigma^{2}_n \it{I}]^{-1} \mathbf{y} \label{eq:posterior:mean} \\
		& cov(\mathbf{f_*}) = K(X_*,X_*) - K(X_*,X)[K(X,X) + \sigma^{2}_n \it{I}]^{-1}K(X,X_*) 
	
	\end{align}
$$

<br/>

For an in depth treatment of _Gaussian Processes_ refer to the [book](https://books.google.nl/books/about/Gaussian_Processes_for_Machine_Learning.html?id=vWtwQgAACAAJ&hl=en).

<br/>

## Modelling Power Plant Outputs

### Drum pressure PSI

```scala
AbottPowerPlant(new PolynomialKernel(2, 0.49), new DiracKernel(0.09),
opt = Map("globalOpt" -> "GS", "grid" -> "4", "step" -> "0.004"),
num_training = 200, num_test = 1000, deltaT = 2, column = 5)
```

<br/>

![water level]({{ site.url }}/public/drum-pressure.png)

<br/>



### Excess Oxygen in exhaust gases (as %)

```scala
AbottPowerPlant(new PolynomialKernel(2, 0.49), new DiracKernel(0.09),
opt = Map("globalOpt" -> "GS", "grid" -> "4", "step" -> "0.004"),
num_training = 200, num_test = 1000, deltaT = 2, column = 6)
```

<br/>

![water level]({{ site.url }}/public/excess-oxygen.png)

<br/>

### Level of water in the drum

```scala
AbottPowerPlant(new PolynomialKernel(2, 0.49), new DiracKernel(0.09),
opt = Map("globalOpt" -> "GS", "grid" -> "4", "step" -> "0.004"),
num_training = 200, num_test = 1000, deltaT = 2, column = 7)
```

<br/>

![water level]({{ site.url }}/public/water-level.png)

<br/>

### Steam Flow Kg./s

```scala
AbottPowerPlant(new PolynomialKernel(2, 0.49), new DiracKernel(0.09),
opt = Map("globalOpt" -> "GS", "grid" -> "4", "step" -> "0.004"),
num_training = 200, num_test = 1000,
deltaT = 2, column = 8)
```

<br/>

![water level]({{ site.url }}/public/steam-flow.png)

<br/>


## Source Code

Below is the example program as a github gist, to view the original program in DynaML, click [here](https://github.com/mandar2812/DynaML/blob/master/src/main/scala/io/github/mandar2812/dynaml/examples/AbottPowerPlant.scala).

{% gist mandar2812/d80a89d5f8edc64dc117 %}
