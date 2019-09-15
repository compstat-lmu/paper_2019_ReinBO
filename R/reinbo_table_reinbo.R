#' @title reinbo
#' @description
#' automatic machine learning with bayesian optimization embedded reinforcement learning
#' @param task an mlr task, see documentations of mlr on how to construct an mlr task
#' @param custom_operators default to NULL, should be specified in a way like list(preprocess = c("cpoScale()", "cpoSpatialSign()", "NA"), filter = NULL, # using all filtering operators, classifier = c("classif.kknn", "classif.naiveBayes")), where "NA" refers to No operation at each stage, but not allowed at the "classifier" stage
#' @param budget how many pipelines to be evaluated
#' @param train_set a vector of indices of the input task used for optimization of pipelines
#' @return best model, best_model$mmodel is the pipeline found, together with the best hyper-parameters, best_model$y is the corresponding negative mmce (mean misclassification error)
#' @export
reinbo = function(task, custom_operators, budget, train_set) {
  ## Parameters for RL environment:
  ctrl = list()
  ctrl$g_operators = g_getOperatorList(custom_operators)
  ctrl$g_max_depth = length(ctrl$g_operators)                  # stages: Scaling --> Feature filtering --> Classification
  ctrl$g_act_cnt = max(sapply(ctrl$g_operators, length))       # max number of available operators at each stage
  ctrl$g_state_names = g_genStateList(ctrl$g_operators)
  ctrl$g_state_dim = length(ctrl$g_state_names)
  ## Parameters for BO_PROBE:
  ctrl$g_init_design = 4   # initial design size for MBO: g_init_design*sum(getParamLengths(par.set))
  ctrl$g_mbo_iter = 2      # iterations of MBO in each episode: g_mbo_iter*sum(getParamLengths(ps))

  conf = rlR::getDefaultConf("AgentTable")
  conf$set(policy.maxEpsilon = 1, policy.minEpsilon = 0.01, policy.aneal.steps = 60)
  best_model = opt.reinbo.table(task, budget = budget, measure = list(mmce), train_set = train_set, init_val = -1, conf = conf, ctrl)
  best_model
}
