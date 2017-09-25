---
title: "Boston Housing Data: Gaussian Process Regression Models"
description: Tutorial about GP regression on housing data using DynaML
date: 2016-03-02
layout: post
categories: posts
tags: gaussian-processes machine-learning boston-housing composite-kernels
comments: True
permalink: /posts/gp-housing/
---

------


## Boston Housing Data

![Boston: Representative Image]({{ site.url }}/public/boston-housing.jpg)

The _Housing_ data set is a popular regression benchmarking data set hosted on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing). It contains 506 records consisting of multivariate data attributes for various real estate zones and their housing price indices. The task is then to learn a regression model that can predict the price index or range. In this blog post, I use the [DynaML](https://github.com/mandar2812/DynaML) machine learning library to train the _GP_ models.

The following meta-data is taken directly from the UCI repository, the final column indicating the property value.

### Attribute Information:

1. **CRIM**: per capita crime rate by town 
2. **ZN**: proportion of residential land zoned for lots over 25,000 sq.ft. 
3. **INDUS**: proportion of non-retail business acres per town 
4. **CHAS**: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
5. **NOX**: nitric oxides concentration (parts per 10 million) 
6. **RM**: average number of rooms per dwelling 
7. **AGE**: proportion of owner-occupied units built prior to 1940 
8. **DIS**: weighted distances to five Boston employment centres 
9. **RAD**: index of accessibility to radial highways 
10. **TAX**: full-value property-tax rate per $10,000 
11. **PTRATIO**: pupil-teacher ratio by town 
12. **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
13. **LSTAT**: % lower status of the population 
14. **MEDV**: Median value of owner-occupied homes in $1000's

## Gaussian Process

Given the lack of data volume (~500 instances) with respect to the dimensionality of the data (13), it makes sense to try smoothing or non-parametric models to model the unknown price function. For a detailed introduction to _Gaussian Processes_, refer to the famous [book](https://books.google.nl/books/about/Gaussian_Processes_for_Machine_Learning.html?id=vWtwQgAACAAJ&hl=en) by Ramussen and Williams. For a succint introduction, you can also refer to the DynaML [wiki](https://github.com/mandar2812/DynaML/wiki/Gaussian-Processes) pages.

$$
	\begin{align}
		& MEDV(\mathbf{u}) = f(\mathbf{u}) + \epsilon(\mathbf{u}) \\
		& f \sim \mathcal{GP}(m(\mathbf{u}), K(\mathbf{u},\mathbf{v})) \\ 
		& \mathbb{E}[\epsilon(\mathbf{u}).\epsilon(\mathbf{v})] = K_{noise}(\mathbf{u}, \mathbf{v})\\
	\end{align}
$$

## Modelling Experiments

In the `examples` folder of the DynaML repository, a program called `TestGPHousing.scala` can be used to test _GP_ models with various kernels, a typical call to `TestGPHousing` looks like.


```scala
TestGPHousing(kernel = new ..., noise = new ..., grid = 10,
step = 0.03, globalOpt = "GS", trainFraction = 0.45)
```

### Kernels

#### FBM kernel with Gaussian Covariate noise


$$
	\begin{align}
		& K(\mathbf{u},\mathbf{v}) = K_{fbm}(\mathbf{u},\mathbf{v}) \\
		& K_{noise} = K_{se}(\mathbf{u},\mathbf{v}) \\
		& K_{se}(\mathbf{u},\mathbf{v}) =  2h.e^{-\frac{1}{2}||\mathbf{u}-\mathbf{v}||^2/\sigma^2} \\
		& K_{fbm}(\mathbf{u},\mathbf{v}) = ||\mathbf{u}||^{2H} + ||\mathbf{v}||^{2H} - ||\mathbf{u}-\mathbf{v}||^{2H} \\
	\end{align}
$$


```scala
DynaML>TestGPHousing(kernel = new FBMKernel(0.55),
noise = new SEKernel(1.5, 1.5), grid = 5,
step = 0.03, globalOpt = "GS", trainFraction = 0.45)
```

```
16/03/03 20:17:42 INFO GridSearch: Optimum value of energy is: 246.38482492249904
Configuration: Map(hurst -> 0.52, bandwidth -> 1.35, amplitude -> 1.35)
16/03/03 20:17:42 INFO SVMKernel$: Constructing kernel matrix.
16/03/03 20:17:42 INFO SVMKernel$: Dimension: 227 x 227
```

```
16/03/03 20:17:43 INFO GPRegression: Generating error bars
16/03/03 20:17:43 INFO RegressionMetrics: Regression Model Performance: MEDV
16/03/03 20:17:43 INFO RegressionMetrics: ============================
16/03/03 20:17:43 INFO RegressionMetrics: MAE: 5.804371810611489
16/03/03 20:17:43 INFO RegressionMetrics: RMSE: 7.676433880135313
16/03/03 20:17:43 INFO RegressionMetrics: RMSLE: 0.4108750385573816
16/03/03 20:17:43 INFO RegressionMetrics: R^2: 0.3713246243782846
16/03/03 20:17:43 INFO RegressionMetrics: Corr. Coefficient: 0.7700074003860581
16/03/03 20:17:43 INFO RegressionMetrics: Model Yield: 0.7243148481557278
16/03/03 20:17:43 INFO RegressionMetrics: Std Dev of Residuals: 6.289145946687416
```
<br/>
![FBM-SE]({{ site.url }}/public/fbm-SE.png)

<br/><br/>


#### Composite FBM + Laplacian Kernel with Uncorrelated Gaussian Noise

$$
	\begin{align}
		& K(\mathbf{u},\mathbf{v}) = K_{lap}(\mathbf{u},\mathbf{v}) + K_{fbm}(\mathbf{u},\mathbf{v}) \\
		& K_{noise}(\mathbf{u},\mathbf{v}) = \delta(\mathbf{u},\mathbf{v}) \\
		& K_{lap}(\mathbf{u},\mathbf{v}) =  e^{-||\mathbf{u}-\mathbf{v}||/\beta} \\
		& K_{fbm}(\mathbf{u},\mathbf{v}) = ||\mathbf{u}||^{2H} + ||\mathbf{v}||^{2H} - ||\mathbf{u}-\mathbf{v}||^{2H} \\
	\end{align}
$$


```scala
DynaML>TestGPHousing(kernel = new FBMKernel(0.55) +
new LaplacianKernel(2.5), noise = new RBFKernel(1.5),
grid = 5, step = 0.03, globalOpt = "GS", trainFraction = 0.45)
```

```
16/03/03 20:45:41 INFO GridSearch: Optimum value of energy is: 278.1603309851301
Configuration: Map(hurst -> 0.4, beta -> 2.35, bandwidth -> 1.35)
16/03/03 20:45:41 INFO SVMKernel$: Constructing kernel matrix.
```

```
16/03/03 20:45:42 INFO GPRegression: Generating error bars
16/03/03 20:45:42 INFO RegressionMetrics: Regression Model Performance: MEDV
16/03/03 20:45:42 INFO RegressionMetrics: ============================
16/03/03 20:45:42 INFO RegressionMetrics: MAE: 5.800070254265218
16/03/03 20:45:42 INFO RegressionMetrics: RMSE: 7.739266267762397
16/03/03 20:45:42 INFO RegressionMetrics: RMSLE: 0.4150438478412412
16/03/03 20:45:42 INFO RegressionMetrics: R^2: 0.3609909626630624
16/03/03 20:45:42 INFO RegressionMetrics: Corr. Coefficient: 0.7633838930006132
16/03/03 20:45:42 INFO RegressionMetrics: Model Yield: 0.7341944950376289
16/03/03 20:45:42 INFO RegressionMetrics: Std Dev of Residuals: 6.287519509352036
```

<br/>

![FBM-SE]({{ site.url }}/public/fbm-lap.png)

<br/>

## Source Code

Below is the example program as a github gist, to view the original program in DynaML, click [here](https://github.com/mandar2812/DynaML/blob/master/src/main/scala/io/github/mandar2812/dynaml/examples/TestGPHousing.scala).

{% gist mandar2812/bc5ff898ca921f22b5ee %}
