---
title: "System Identification using LSSVMs: Pont-sur-Sambre Power Plant"
description: Tutorial about LSSVM based NARX models on Daisy power plant data using DynaML
date: 2016-03-04
layout: post
categories: posts
tags: lssvm machine-learning system-identification kernel-methods
comments: True
permalink: /posts/lssvm-power-plant/
---

## System Identification

_System identification_ refers to the process of learning a predictive model for a given dynamic system i.e. a system whose dynamics evolve with time. The most important aspect of these models is their structure, specifically the following are the common dynamic system models for discretely sampled time dependent systems.

### Nonlinear AutoRegresive (NAR)
Signal $$y(t)$$ modeled as a function of its previous $$p$$ values

$$
	\begin{align}
    y(t) = f(y(t-1), y(t-2), \cdots, y(t-p)) + \epsilon(t)
	\end{align}
$$

### Nonlinear AutoRegressive with eXogenous inputs (NARX)
Signal $$y(t)$$ modeled as a function of the previous $$p$$ values of itself and the $$m$$ exogenous inputs $$u_{1}, \cdots u_{m}$$

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

## DaISy: System Identification Database ([link](http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html))

_DaISy_ is a database of (artificial and real world) dynamic systems maintained by the [STADIUS](https://www.esat.kuleuven.be/stadius/) research group at KU Leuven. We will work with the power plant data set listed on the _DaISy_ home page in this post. Using [DynaML](https://tailhq.github.io/DynaML), which comes preloaded with the power plant data, we will train [LSSVM](https://github.com/tailhq/DynaML/wiki/Dual-LSSVM) models to predict the various output indicators of the power plant in question.

### Pont-sur-Sambre Power Plant Data ([link](ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/powerplant.txt))

![Pont-sur-Sambre: Representative Image]({{ site.url }}/public/powerplant.jpg)

#### Attributes

**Instances**: 200

**Inputs**:

1. Gas flow
2. Turbine valves opening
3. Super heater spray flow
4. Gas dampers
5. Air flow

**Outputs**:
6. Steam pressure
7. Main stem temperature
8. Reheat steam temperature

## System Modelling

An LSSVM _NARX_ of autoregressive order $$p = 2$$ is chosen to model the plant output data. An LSSVM model builds a predictor of the following form.

$$
	\begin{align*}
	y(x) = \sum_{k = 1}^{N}\alpha_k K(\mathbf{x}, \mathbf{x_k}) + b
	\end{align*}
$$

Which is the result of solving the following linear system. 

$$
	\left[\begin{array}{c|c}
   0  & 1^\intercal_v   \\ \hline
   1_v & K + \gamma^{-1} \mathit{I} 
\end{array}\right] 
\left[\begin{array}{c}
   b    \\ \hline
   \alpha  
\end{array}\right] = \left[\begin{array}{c}
   0    \\ \hline
   y  
\end{array}\right]
$$

Here the matrix $$K$$ is constructed from the training data using a kernel function $$ K(\mathbf{x}, \mathbf{y}) $$.

### Choice of Kernel Function

For this problem we choose a polynomial kernel.

$$
	\begin{align*}
		K(\mathbf{x},\mathbf{y}) = K_{poly}(\mathbf{x},\mathbf{y}) = (\mathbf{x}^{T}.\mathbf{y} + \alpha)^{d}
	\end{align*}
$$


### Steam Pressure

```scala
DynaML>DaisyPowerPlant(new PolynomialKernel(2, 0.5),
opt = Map("regularization" -> "2.5", "globalOpt" -> "GS",
"grid" -> "4", "step" -> "0.1"),
num_training = 100, deltaT = 2,
column = 6)
```

```shell
16/03/04 17:13:43 INFO RegressionMetrics: Regression Model Performance: steam pressure
16/03/04 17:13:43 INFO RegressionMetrics: ============================
16/03/04 17:13:43 INFO RegressionMetrics: MAE: 82.12740530161123
16/03/04 17:13:43 INFO RegressionMetrics: RMSE: 104.39251587470388
16/03/04 17:13:43 INFO RegressionMetrics: RMSLE: 0.9660077848586197
16/03/04 17:13:43 INFO RegressionMetrics: R^2: 0.8395534877128238
16/03/04 17:13:43 INFO RegressionMetrics: Corr. Coefficient: 0.9311734118932473
16/03/04 17:13:43 INFO RegressionMetrics: Model Yield: 0.6288000962818303
16/03/04 17:13:43 INFO RegressionMetrics: Std Dev of Residuals: 87.82754320038951
```

![Steam Pressure]({{ site.url }}/public/steampressure.png)

![Steam Pressure]({{ site.url }}/public/steampressure-fit.png)

### Reheat Steam Temperature

```scala
DaisyPowerPlant(new PolynomialKernel(2, 1.5),
opt = Map("regularization" -> "2.5", "globalOpt" -> "GS",
"grid" -> "4", "step" -> "0.1"), num_training = 150,
deltaT = 1, column = 8)
```

```shell
16/03/04 16:50:42 INFO RegressionMetrics: Regression Model Performance: reheat steam temperature
16/03/04 16:50:42 INFO RegressionMetrics: ============================
16/03/04 16:50:42 INFO RegressionMetrics: MAE: 124.60921194767073
16/03/04 16:50:42 INFO RegressionMetrics: RMSE: 137.33314302068544
16/03/04 16:50:42 INFO RegressionMetrics: RMSLE: 0.5275727128626408
16/03/04 16:50:42 INFO RegressionMetrics: R^2: 0.8247581957573777
16/03/04 16:50:42 INFO RegressionMetrics: Corr. Coefficient: 0.9744133881055823
16/03/04 16:50:42 INFO RegressionMetrics: Model Yield: 0.7871288689840381
16/03/04 16:50:42 INFO RegressionMetrics: Std Dev of Residuals: 111.86852905896446
```

![Steam Temp]({{ site.url }}/public/temperature.png)

![Steam Temp]({{ site.url }}/public/temperature-fit.png)

<br/>

## Source Code

Below is the example program as a github gist, to view the original program in DynaML, click [here](https://github.com/tailhq/DynaML/blob/master/src/main/scala/io/github/mandar2812/dynaml/examples/DaisyPowerPlant.scala).

{% gist mandar2812/eb23b47adad66deb2f65 %}
