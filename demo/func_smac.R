run = function(cs, budget = 1000) {
  hh = reticulate::import("python_smac_space")
  #scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
  #                   "runcount-limit": budget,  # maximum function evaluations
  #                   "cs": cs,               # configuration space
  #                   "deterministic": "true"
  #                   })
  budget = 100
  scenario = hh$Scenario(list("run_obj" = "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit" = budget,  # maximum function evaluations
                     "cs" = cs,               # configuration space
                     "deterministic" = "true",
                     "shared_model" = TRUE   # deletable
                     ))

  # scenario$abort_on_first_run_crash = F

  print("Optimizing! Depending on your machine, this might take a few minutes.")
  np = reticulate::import("numpy")
  #fd = hh$ExecuteTAFuncDict(toy_smac_obj)
  #smac = hh$SMAC(scenario = scenario, rng = np$random$RandomState(as.integer(4)), tae_runner = toy_smac_obj)
  #smac = hh$SMAC(scenario = scenario, rng = np$random$RandomState(as.integer(4)), tae_runner = fd)
  reticulate::source_python('smac_obj.py')
  source("smac_obj.R")
  #smac = hh$SMAC(scenario = scenario, rng = np$random$RandomState(as.integer(4)), tae_runner = smac_obj_from_cfg)
  py_fun = reticulate::r_to_py(toy_smac_obj, convert = FALSE)
  #py_fun = reticulate::r_to_py(function(x) 1, convert = TRUE)
  smac = hh$SMAC(scenario = scenario, rng = np$random$RandomState(as.integer(4)), tae_runner = py_fun)
  smac$get_tae_runner()
  incumbent = smac$optimize()  # problem
  #inc_value = svm_from_cfg(incumbent)
  incumbent
  #print("Optimized Value: %.2f" % (inc_value))
}


test_run = function() {
  cfg = reticulate::import("python_smac_space")
  cs = cfg$cs
  run(cs)
}

# Predict function: evaluate best model on test dataset
lock_eval.smac = function(task, measure, train_set, test_set, best_model){
  cfg = best_model
  lrn = gen_mlrCPOPipe_from_smac_cfg(cfg)
  mod = train(lrn, task, subset = train_set)
  pred = predict(mod, task, subset = test_set)
  mpred = performance(pred, measures = measure)
  return(mpred)
}


gen_mlrCPOPipe_from_smac_cfg = function(cfg) {
    #cfg = cfg.sample_configuration()
    # convert ConfigSpace.configuration_space.ConfigurationSpace to ConfigSpace.configuration_space.Configuration
    # For deactivated parameters, the configuration stores None-values. so we remove them.
    #cfg = list(Model = "xgboost", Preprocess = "cpoScale(center = FALSE)", FeatureFilter = "cpoPca(center = FALSE, rank = rank_val)", lrn_xgboost_max_depth = 3, lrn_xgboost_eta = 0.03, fe_pca_rank = 0.5) # for testing and debug
    model = cfg$Model
    preprocess = cfg$Preprocess
    pfilter = cfg$FeatureFilter
    perc_val = NULL
    rank_val = NULL

    ##
    extract_hyper_prefix = function(prefix = "lrn", cfg) {
      names4lrn_hyp = grep(pattern = prefix, x = names(cfg), value = T)
      ps.learner = cfg[names4lrn_hyp]  # evaluted later by R function eval
      pattern = paste0("(", prefix, "_[:alpha:]+_)*")
      #ns4hyper = gsub(pattern = pattern, x = names4lrn_hyp, replacement="", ignore.case = T)
      ns4hyper = stringr::str_replace(string = names4lrn_hyp, pattern = pattern, replacement="")
      names(ps.learner) = ns4hyper
      ps.learner
    }
    ##
    ps.learner  =  extract_hyper_prefix("lrn", cfg)  # hyper-parameters for learner must exist

    names4Fe = grep(pattern = "fe", x = names(cfg), value = T)

    p = mlr::getTaskNFeats(subTask)  # this subTask relies on global variable

    if(length(names4Fe) > 0) {
      ps.Fe = extract_hyper_prefix("fe", cfg)
      if(grepl(pattern = "perc", x = names(ps.Fe)))  {
        name4featureEng_perc = grep(pattern = "perc", x = names(ps.Fe), value = T)
        perc_val = ps.Fe[[name4featureEng_perc]] 
      }
      if(grepl(pattern = "rank", x = names(ps.Fe))) {
        name4featureEng_rank = grep(pattern = "rank", x = names(ps.Fe), value = T)
        rank_val = ceiling(ps.Fe[[name4featureEng_rank]] * p)
      }
    }

    lrn = sprintf("%s %%>>%% %s %%>>%% makeLearner('classif.%s', par.vals = ps.learner)",
                preprocess, pfilter, model)
    lrn = gsub(pattern = "NA %>>%", x = lrn, replacement = "", fixed = TRUE)

 
    # set mtry after reducing the number of dimensions
    if (model == "ranger") {
        p1 = p
        if (!is.null(perc_val)) {p1 = max(1, round(p*perc_val))}
        if (!is.null(rank_val)) {p1 = rank_val}
        ps.learner$mtry = max(1, as.integer(p1*ps.learner$mtry))
    }
    lrn = paste0("library(mlrCPO);library(magrittr);", lrn)
    obj_lrn = eval(parse(text = lrn))
    return(obj_lrn)
}

test_gen_mlrCPOPipe_from_smac_cfg = function() {
  subTask = mlr::iris.task
  cfg = reticulate::import("python_smac_space")
  cfg = cfg$stub
  lrn = gen_mlrCPOPipe_from_smac_cfg(cfg)
  lrn
}
