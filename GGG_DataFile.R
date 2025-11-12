# Libraries --------------------------------------------------------------
library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(discrim)
setwd("~/GitHub/GGG_Comptetion")

# Read files --------------------------------------------------------------
train_data <- vroom("train.csv")
test_data  <- vroom("test.csv")

# Make sure your outcome column is a factor
train_data <- train_data %>%
  mutate(ACTION = factor(type)) %>%
  select(-type)   # remove duplicate predictor

# Recipe ------------------------------------------------------------------
my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_lencode(all_nominal_predictors(), outcome = vars(ACTION), smooth = FALSE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


# Random forest model -----------------------------------------------------
rf_mod <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = 1000
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

folds <- vfold_cv(train_data, v = 5, strata = ACTION)

rf_grid <- grid_regular(
  mtry(range = c(1, 6)),
  min_n(range = c(5, 25)),
  levels = 5
)

rf_tuned <- tune_grid(
  rf_wf,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc, accuracy)
)

rf_best <- select_best(rf_tuned, metric = "roc_auc")
rf_final_wf <- finalize_workflow(rf_wf, rf_best)
rf_final_fit <- fit(rf_final_wf, data = train_data)

rf_predictions <- predict(rf_final_fit, new_data = test_data, type = "class") %>%
  rename(type = .pred_class)

kaggle_rf_submission <- test_data %>%
  select(id) %>%
  bind_cols(rf_predictions)

vroom_write(
  kaggle_rf_submission,
  file = "./Forest.csv",
  delim = ","
)



# Bayes -------------------------------------------------------------------

# Naive Bayes model (fixed smoothing)
nb_mod <- naive_Bayes(
  smooth = 1.5  # Laplace smoothing
) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

# Workflow
nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_mod)

# Fit final model
nb_final_fit <- fit(nb_wf, data = train_data)

# Predict classes for Kaggle
nb_predictions <- predict(nb_final_fit, new_data = test_data, type = "class") %>%
  rename(type = .pred_class)

# Prepare Kaggle submission
kaggle_nb_submission <- test_data %>%
  select(id) %>%
  bind_cols(nb_predictions)

vroom_write(
  kaggle_nb_submission,
  file = "./NaiveBayes.csv",
  delim = ","
)



# # XG Boost ----------------------------------------------------------------
# 
# 
# xgb_mod <- boost_tree(
#   trees = 10000,            # number of boosting rounds
#   tree_depth = tune(),
#   learn_rate = tune(),
#   loss_reduction = tune(),  # min split loss (gamma)
#   sample_size = tune(),     # row subsample
#   mtry = tune(),            # number of predictors sampled
#   stop_iter = 50            # early stopping rounds
# ) %>%
#   set_engine("xgboost") %>%
#   set_mode("classification")
# 
# # Workflow ---------------------------------------------------------------
# xgb_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(xgb_mod)
# 
# # Cross-validation -------------------------------------------------------
# set.seed(123)
# folds <- vfold_cv(train_data, v = 5, strata = ACTION)
# 
# # Tuning grid ------------------------------------------------------------
# xgb_grid <- grid_latin_hypercube(
#   tree_depth(range = c(3, 10)),
#   learn_rate(range = c(0.01, 0.3)),
#   loss_reduction(range = c(0, 5)),
#   sample_size = sample_prop(range = c(0.5, 1)),
#   finalize(mtry(), train_data),
#   size = 20
# )
# 
# # Tune model -------------------------------------------------------------
# xgb_tuned <- tune_grid(
#   xgb_wf,
#   resamples = folds,
#   grid = xgb_grid,
#   metrics = metric_set(roc_auc, accuracy)
# )
# 
# # Select best model ------------------------------------------------------
# xgb_best <- select_best(xgb_tuned, metric = "accuracy")
# 
# # Finalize workflow ------------------------------------------------------
# xgb_final_wf <- finalize_workflow(xgb_wf, xgb_best)
# 
# # Fit final model --------------------------------------------------------
# xgb_final_fit <- fit(xgb_final_wf, data = train_data)
# 
# # Predict on test data ---------------------------------------------------
# test_predictions <- predict(xgb_final_fit, new_data = test_data, type = "class")
# 
# # Make a clean submission tibble
# kaggle_submission <- bind_cols(
#   id = test_data$id,
#   type = test_predictions$.pred_class  # Use the original 'type' column name
# )
# 
# # Write CSV
# vroom_write(
#   kaggle_submission,
#   file = "./XGmod_class.csv",
#   delim = ","
# )
# 
