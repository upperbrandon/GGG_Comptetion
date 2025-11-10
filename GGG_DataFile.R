# Libraries --------------------------------------------------------------
library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)

# Set working directory
setwd("~/GitHub/GGG_Comptetion")

# Read files --------------------------------------------------------------
train_data <- vroom("train.csv")
test_data  <- vroom("test.csv")

# Make sure your outcome column is a factor
train_data <- train_data %>%
  mutate(ACTION = factor(type))  # <-- confirm "type" exists

# Recipe ------------------------------------------------------------------
my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  update_role(type, new_role = "id") %>%
  update_role_requirements(role = "id", bake = FALSE) %>%  # <— fix here
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_lencode(all_nominal_predictors(), outcome = vars(ACTION), smooth = FALSE) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

pg <- prep(my_recipe)
train_processed <- bake(pg, new_data = NULL)
test_processed  <- bake(pg, new_data = test_data)
