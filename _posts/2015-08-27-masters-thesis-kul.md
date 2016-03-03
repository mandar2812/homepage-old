---
title: "Masters Thesis: ESAT, KU Leuven"
description: KU Leuven, ESAT, MAI Thesis Defense
date: 2015-08-27
layout: pdf
categories: papers
tags: svm,machine-learning
file: Masters_Thesis_Mandar.pdf
permalink: /papers/mai-thesis/
---

### Fixed Size Least Squares Support Vector Machines: A Scala based programming framework for Large Scale Classification


### Abstract


We propose _FS-Scala_, a flexible and modular _Scala_ based implementation of the Fixed Size Least Squares Support Vector Machine (FS-LSSVM) for large data sets. The framework consists of a set of modules for (gradient and gradient free) optimization, model representation, kernel functions and evaluation of FS-LSSVM models.

A kernel based _Fixed-Size Least Squares Support Vector Machine_ (FS-LSSVM) model is implemented in the proposed framework, while heavily employing distributed _MapReduce_ via the parallel computing capabilities of _Apache Spark_. Global optimization routines like _Coupled Simulated Annealing_ (CSA) and _Grid Search_ are implemented and used to tune the hyper-parameters of the FS-LSSVM model.

Finally, we carry out experiments on benchmark data sets like _Forest Cover Type_, _Magic Gamma_ and _Adult_, recording the performance and tuning time of various kernel based FS-LSSVM models.

### FS-LSSVM: Formulation


$$
\begin{equation}
\label{eqfs}
\min_{w,b} \mathcal{J}(w,b) \ = \ \frac{1}{2}w^{\intercal} w + \frac{\gamma}{2}\sum^{n}_{i=1} \left(y_{i} - w^{\intercal} \hat{\phi}(x_i) - b\right)^{2}.
\end{equation}
$$

The solution of which is given by:

$$
\begin{align}
\label{eqfssol}
& \left( \begin{matrix}
\hat{w}\\ 
\hat{b}
\end{matrix}\right ) = 
\left ( \hat{\Phi}^{\intercal}_e \hat{\Phi}_e + \frac{\mathit{I}_{m+1}}{\gamma} \right )^{-1} \hat{\Phi}^{\intercal}_e y,
\\ \nonumber \\
\text{where} \hspace{10pt}
& \hat{\Phi}_e = \begin{pmatrix}
\hat{\phi}_{1}(x_1) & \cdots & \hat{\phi}_{m}(x_1) & 1\\ 
\vdots &  \ddots & \vdots & \vdots\\ 
\hat{\phi}_{1}(x_n) & \cdots & \hat{\phi}_{m}(x_n) & 1
\end{pmatrix}. \nonumber
\end{align}
$$

### Citation
    
    @mastersthesis{
        author = {Chandorkar, M. H.},
        title = {Fixed Size Least Squares Support Vector Machines:
		A Scala based programming framework for Large Scale Classification},
        school = {Katholieke Universitiet Leuven},
        year = {2015}
    }


### Link

_FS-Scala_ can be found [here](https://github.com/mandar2812/FS-Scala).

