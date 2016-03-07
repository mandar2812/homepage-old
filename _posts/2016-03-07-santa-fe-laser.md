---
title: "System Identification using Gaussian Processes: Santa Fe Laser Data Set"
description: Tutorial about GP based NAR models on Santa Fe laser data using DynaML
date: 2016-03-07
layout: post
categories: posts
tags: gaussian-process,machine-learning,system-identification,kernels,santa-fe-laser
comments: True
permalink: /posts/santa-fe-laser/
---

## System Identification
For a short introduction to system identification and some common models refer to this previous [post](/posts/lssvm-power-plant). Below I give a short tour of the _Santa Fe Laser_ example which comes shipped with the [DynaML](https://github.com/mandar2812/DynaML) machine learning library.

## Santa Fe Laser Generated Data 

![Santa Fe]({{ site.url }}/public/SantaFe.png)

The Santa Fe laser data is a standard benchmark data set in system identification. It serves as good starting point to start exploring time series models. It records only one observable (laser intensity), has little noise and is generated from a known physics dynamical process. A more detailed explanation is given below.

>The measurements were made on an 81.5-micron 14NH3 cw (FIR) laser, pumped optically by the P(13) line of an N2O laser via the vibrational aQ(8,7) NH3 transition. The basic laser setup can be found in Ref. 1. The intensity data was recorded by a LeCroy oscilloscope. No further processing happened. The experimental signal to noise ratio was about 300 which means slightly under the half bit uncertainty of the analog to digital conversion. The data is a cross-cut through periodic to chaotic intensity pulsations of the laser. Chaotic pulsations more or less follow the theoretical Lorenz model (see References) of a two level system.
>
>[Source](http://www-psych.stanford.edu/~andreas/Time-Series/SantaFe.html)

## Santa Fe Laser: NAR model

The data set is unidimensional, so we can only train a Nonlinear Auto-Regressive (NAR) model for the laser intensity. Choosing the auto-regressive order $$ p = 2 $$, we train two candidate _NAR_ models.

### Choice of Kernel Function

For this problem we build models based on two kernels.

1. Radial Basis Function (RBF):
$$
	\begin{align}
		& K(\mathbf{u},\mathbf{v}) = K_{rbf}(\mathbf{u},\mathbf{v}) \\
		& K_{rbf}(\mathbf{u},\mathbf{v}) =  \frac{1}{2}e^{-\frac{1}{2}||\mathbf{u}-\mathbf{v}||^2/\sigma^2} \\
		& K_{noise} = \delta(\mathbf{u},\mathbf{v}) \\
	\end{align}
$$

```scala
SantaFeLaser(new RBFKernel(2.5), new DiracKernel(1.0),
opt = Map("globalOpt" -> "GS", "grid" -> "10", "step" -> "0.1"),
num_training = 200, num_test = 500, deltaT = 5)
```

```
16/03/07 22:03:10 INFO RegressionMetrics: Regression Model Performance: Laser Intensity
16/03/07 22:03:10 INFO RegressionMetrics: ============================
16/03/07 22:03:10 INFO RegressionMetrics: MAE: 10.919757407593648
16/03/07 22:03:10 INFO RegressionMetrics: RMSE: 18.527723082632765
16/03/07 22:03:10 INFO RegressionMetrics: RMSLE: 0.41343485025397475
16/03/07 22:03:10 INFO RegressionMetrics: R^2: 0.8550953005807426
16/03/07 22:03:10 INFO RegressionMetrics: Corr. Coefficient: 0.928916961722154
16/03/07 22:03:10 INFO RegressionMetrics: Model Yield: 0.6597256758459964
16/03/07 22:03:10 INFO RegressionMetrics: Std Dev of Residuals: 18.1615168822832
```

![Steam Fe]({{ site.url }}/public/santa-fe-pred.png)

![Santa Fe]({{ site.url }}/public/santa-fe-rbf-fit.png)


2. Fractional Brownian Field (FBM):
$$
	\begin{align}
		& K(\mathbf{u},\mathbf{v}) = K_{fbm}(\mathbf{u},\mathbf{v}) \\
		& K_{fbm}(\mathbf{u},\mathbf{v}) = ||\mathbf{u}||^{2H} + ||\mathbf{v}||^{2H} - ||\mathbf{u}-\mathbf{v}||^{2H} \\
		& K_{noise} = \delta(\mathbf{u},\mathbf{v}) \\
	\end{align}
$$

```scala
DynaML>SantaFeLaser(new FBMKernel(1.1), new DiracKernel(1.0),
opt = Map("globalOpt" -> "GS", "grid" -> "10", "step" -> "0.1"),
num_training = 200, num_test = 500, deltaT = 2)
```

```
16/03/07 22:07:46 INFO RegressionMetrics: Regression Model Performance: Laser Intensity
16/03/07 22:07:46 INFO RegressionMetrics: ============================
16/03/07 22:07:46 INFO RegressionMetrics: MAE: 8.466099689528546
16/03/07 22:07:46 INFO RegressionMetrics: RMSE: 13.523138654434868
16/03/07 22:07:46 INFO RegressionMetrics: RMSLE: 0.38303731310173433
16/03/07 22:07:46 INFO RegressionMetrics: R^2: 0.9228042537204268
16/03/07 22:07:46 INFO RegressionMetrics: Corr. Coefficient: 0.964525269647539
16/03/07 22:07:46 INFO RegressionMetrics: Model Yield: 0.7656581073289345
16/03/07 22:07:46 INFO RegressionMetrics: Std Dev of Residuals: 14.742253950108552
```

![Steam Fe]({{ site.url }}/public/santafe-pred.png)

![Santa Fe]({{ site.url }}/public/santafe-fit.png)

## Source Code

{% gist mandar2812/0ac7ea02b73548c2e61d %}
