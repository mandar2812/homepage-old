---
title: "Neural Networks: Classification of Wine Quality"
description: Tutorial: Training Neural Network classifiers on the wine quality data, using DynaML
date: 2016-03-30
layout: post
categories: posts
tags: neural-networks,machine-learning,classification,DynaML,wine-quality
comments: True
permalink: /posts/nn-wine-quality/
---

------

The [_wine quality_](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) data set is a common example used to benchmark classification models. Here we use the [DynaML](mandar2812.github.io/DynaML) scala machine learning environment to train classifiers to detect 'good' wine from 'bad' wine. A short listing of the data attributes/columns is given below. The UCI archive has two files in the wine quality data set namely ```winequality-red.csv``` and ```winequality-white.csv```. We train two separate classification models, one for red wine and one for white. 

![Wine: Representative Image]({{ site.url }}/public/wine.jpg)


## Data Set

**Inputs**:

1. fixed acidity 
2. volatile acidity 
3. citric acid 
4. residual sugar 
5. chlorides 
6. free sulfur dioxide 
7. total sulfur dioxide 
8. density 
9. pH 
10. sulphates 
11. alcohol 

**Output** (based on sensory data): 
12. quality (score between 0 and 10)

### Data Output Preprocessing

The wine quality target variable can take integer values from `0` to `10`, first we convert this into a binary class variable by setting the quality to be 'good'(encoded by the value `1`) if the numerical value is greater than `6` and 'bad' (encoded by value `-1.0`) otherwise.

## Wine Quality: Neural Network Experiment

The ```TestNNWineQuality``` program in the DynaML examples package contains all the required code for model building and testing, see the gist below for more details.

### Red Wine

```scala
TestNNWineQuality(hidden = 1, nCounts = List(2),
acts = List("linear", "tansig"), stepSize = 0.2, maxIt = 130,
mini = 1.0, alpha = 0.0,
training = 1000, test = 600,
regularization = 0.001,
wineType = "red")
```

```
16/03/30 18:59:38 INFO BinaryClassificationMetrics: Classification Model Performance: red wine
16/03/30 18:59:38 INFO BinaryClassificationMetrics: ============================
16/03/30 18:59:38 INFO BinaryClassificationMetrics: Accuracy: 0.8566666666666667
16/03/30 18:59:38 INFO BinaryClassificationMetrics: Area under ROC: 0.7782440503121889
16/03/30 18:59:38 INFO BinaryClassificationMetrics: Maximum F Measure: 0.755966787057378
```

![red-roc]({{site.url}}/public/red-wine-roc.png)

![red-fmeasure]({{site.url}}/public/red-wine-fmeasure.png)


### White Wine

```scala
TestNNWineQuality(hidden = 1, nCounts = List(3),
acts = List("linear", "tansig"), stepSize = 0.16, maxIt = 100,
mini = 1.0, alpha = 0.0,
training = 1500, test = 3000,
regularization = 0.001,
wineType = "white")
```

![white-roc]({{site.url}}/public/white-wine-roc.png)

![white-fmeasure]({{site.url}}/public/white-wine-fmeasure.png)


```
16/03/30 18:49:58 INFO BinaryClassificationMetrics: Classification Model Performance: white wine
16/03/30 18:49:58 INFO BinaryClassificationMetrics: ============================
16/03/30 18:49:58 INFO BinaryClassificationMetrics: Accuracy: 0.8096666666666666
16/03/30 18:49:58 INFO BinaryClassificationMetrics: Area under ROC: 0.7784814672924049
16/03/30 18:49:58 INFO BinaryClassificationMetrics: Maximum F Measure: 0.7570286230962675
```

## Source Code

{% gist mandar2812/f918bc0b52ec1b08e5bfe988a5657f9a %}
