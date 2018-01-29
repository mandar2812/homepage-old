---
title: "Logistic Regression: Classification of Wine Quality"
description: "Tutorial: Training a logistic regression model on the wine quality data, using DynaML"
date: 2016-04-01
layout: post
categories: posts
tags: logistic-regression machine-learning classification DynaML wine-quality
comments: True
permalink: /posts/logistic-regression-wine-quality/
---

------

In the previous [post](/posts/nn-wine-quality/), we trained [DynaML](/DynaML/)'s feed forward neural networks on the [wine quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) data set. Lets compare how single layer feed forward neural networks compare to a simple logistic regression trained using [_Gradient Descent_](https://transcendent-ai-labs.github.io/DynaML/core/core_opt_convex/#gradient-descent). The ```TestLogisticWineQuality``` program in the ```examples``` package does precisely that (check out the source code below). 

## Red Wine


```scala
TestLogisticWineQuality(stepSize = 0.2, maxIt = 120,
mini = 1.0, training = 800,
test = 800, regularization = 0.2,
wineType = "red")
```

```
16/04/01 15:21:57 INFO BinaryClassificationMetrics: Classification Model Performance
16/04/01 15:21:57 INFO BinaryClassificationMetrics: ============================
16/04/01 15:21:57 INFO BinaryClassificationMetrics: Accuracy: 0.8475
16/04/01 15:21:57 INFO BinaryClassificationMetrics: Area under ROC: 0.7968417788802267
16/04/01 15:21:57 INFO BinaryClassificationMetrics: Maximum F Measure: 0.7493563745371187
```

![red-roc]({{site.url}}/public/red-wine-logistic-roc.png)

![red-fmeasure]({{site.url}}/public/red-wine-logistic-fmeasure.png)


## White Wine


```scala
TestLogisticWineQuality(stepSize = 0.26, maxIt = 300,
mini = 1.0, training = 3800,
test = 1000, regularization = 0.0,
wineType = "white")
```

```
16/04/01 15:27:17 INFO BinaryClassificationMetrics: Classification Model Performance
16/04/01 15:27:17 INFO BinaryClassificationMetrics: ============================
16/04/01 15:27:17 INFO BinaryClassificationMetrics: Accuracy: 0.829
16/04/01 15:27:17 INFO BinaryClassificationMetrics: Area under ROC: 0.7184782682020251
16/04/01 15:27:17 INFO BinaryClassificationMetrics: Maximum F Measure: 0.7182203962483446
```



![red-roc]({{site.url}}/public/white-wine-logistic-roc.png)

![red-fmeasure]({{site.url}}/public/white-wine-logistic-fmeasure.png)


## Comparison with Neural Networks

Considering that a simple logistic regression model performs quite well on the data, and that logistic regression is equivalent to a single perceptron neural network model, we can train a neural net with `0` hidden layers using the ```TestNNWineQuality``` program.

```scala
TestNNWineQuality(0, List(), List("tansig"), stepSize = 0.2, maxIt = 120, 
mini = 1.0, alpha = 0.0, training = 1200, test = 400, regularization = 0.0, 
wineType = "red")
```

```
16/04/01 14:04:34 INFO BinaryClassificationMetrics: Classification Model Performance
16/04/01 14:04:34 INFO BinaryClassificationMetrics: ============================
16/04/01 14:04:34 INFO BinaryClassificationMetrics: Accuracy: 0.895
16/04/01 14:04:34 INFO BinaryClassificationMetrics: Area under ROC: 0.8209578913532626
16/04/01 14:04:34 INFO BinaryClassificationMetrics: Maximum F Measure: 0.7975192758967482
```

Which gives a performance in the same ball park as the logistic regression model, here it must be noted that, we used a larger training set fraction and a hyperbolic tangent activation function.

## Source Code

{% gist mandar2812/b309d5c26b5aba9c84415d2f7cd6d913 %}
