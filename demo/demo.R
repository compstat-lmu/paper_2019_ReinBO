library(magrittr)
library(mlrCPO)
library(OpenML)
library(ReinBo)
task = convertOMLTaskToMlr(getOMLTask(37))$mlr.task %>>% cpoDummyEncode(reference.cat = FALSE)
outer_loop = makeResampleInstance("CV", iters = 5, stratify = TRUE, task)
train_set = outer_loop$train.inds[[1]]
test_set = outer_loop$test.inds[[1]]


# The pipeline search space could also be customized by setting, e.g.,
custom_operators = list(preprocess = c("cpoScale()", "NA"),
                          filter = c("cpoPca(center = FALSE, rank)", "cpoFilterAnova(perc)", "NA"),
                          classifier = c("classif.kknn", "classif.naiveBayes"))




best_model = reinbo(task, custom_operators = NULL, budget = 100, train_set = train_set)
# best_model = reinbo(task, custom_operators = custom_operators, budget = 100, train_set = train_set)
pred = lock_eval.reinbo.table(task, measure = list(mmce), train_set, test_set, best_model)
best_model$env$agent$q_tab
best_model$env$agent$act_names_per_state
