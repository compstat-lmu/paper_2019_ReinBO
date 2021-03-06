## Introduction
**Reinbo** is an AutoML solution in R that optimizes machine learning pipeline with Bayesian Optimization embedded Reinforcement Learning.

Read our ECML Auto Data Science workshop paper for more information.

ReinBo: Machine Learning pipeline search and configuration with Bayesian Optimization embedded Reinforcement Learning, Xudong Sun and Jiali Lin and Bernd Bischl, arxiv preprint, https://arxiv.org/abs/1904.05381, 2019

For citation, see reinbo_citation.bib in this repository.

# Reproduce the benchmark experiments

The codes to reproduce the experiments in the paper lies in the directory "benchmark". 

- install the required packages through benchmark/install_depend.R
- learn how to use the R cran package batchtools for large scale benchmark study
- in folder benchmark, execute main.R, then submit jobs according to "batchtools" API
- There are in total 600 jobs


## Installation of the Package

We also have an R package developed, to install, first you should have the rlR package ready

install rlR through

```r
devtools::install_github("smilesun/rlR")
```

read the instructions about rlR to make sure it works on your computer.

afterwards,


```r
devtools::install_github("compstat-lmu/paper_2019_ReinBo")
```

## Using ReinBo Package
Load package to library and now ReinBo is ready for you to optimize a pipeline.

```r
library(ReinBo)
best_model = reinbo(task = mlrTask, budget = 1000L, train_set = train_set, custom_operators = NULL)
```
- **task**: the task must be a **mlr task** (currently only classification task is accepted.)
- **budget**: maximum number of pipelines to evaluate
- **train_set**: optimization data set index vector
- **custom_operators**: set **Null** to use all default operators for pipeline

A typical ML pipline consists of 3 stages: preprocessing, filtering and classification. Below is a list of the current built-in operators at each stage that come with ReinBo:

- **preprocess**: "cpoScale()", "cpoScale(scale = FALSE)", "cpoScale(center = FALSE)", "cpoSpatialSign()", "NA";

- **filter**: "cpoFilterAnova(perc)", "cpoFilterKruskal(perc)", "cpoPca(center = FALSE, rank)", "cpoFilterUnivariate(perc)", "NA";

- **classifier**: "classif.ksvm", "classif.ranger", "classif.kknn", "classif.xgboost", "classif.naiveBayes";

where "NA" indicates that no operator would be taken at that stage.

Users can also select a subset of operators by setting e.g.:

```r
custom_operators = list(preprocess = c("cpoScale()", "cpoSpatialSign()", "NA"),
                        filter = NULL, # using all filtering operators
                        classifier = c("classif.kknn", "classif.naiveBayes"))
```

## Example

```r
library(ReinBo)
library(mlrCPO)
library(OpenML)
task = convertOMLTaskToMlr(getOMLTask(37))$mlr.task %>>% cpoDummyEncode(reference.cat = FALSE)
split = makeResampleInstance("Holdout", task)
train_set = split$train.inds[[1]]
test_set = split$test.inds[[1]]
best_model = reinbo(task = task, budget = 100L, train_set = train_set, custom_operators = NULL)
print(best_model$mmodel)
```

```
##                                                  Model        C     sigma
## 13 cpoScale()\tcpoFilterUnivariate(perc)\tclassif.ksvm 6.976259 -7.312171
##         perc          y
## 13 0.5984765 -0.2304947
```

**y** in the result is the negative mmce (mean mis-classification error) of the best model.
