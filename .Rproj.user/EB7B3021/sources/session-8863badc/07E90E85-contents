library(tidymodels)

data <- read.csv("lab_data.csv", header = TRUE)
data$lodgepole_pine <- factor(data$lodgepole_pine)

# using the standard predictive analytics/machine learning approach with the tidymodels framework 
data_split <- initial_split(data, strata = "lodgepole_pine", prop = 0.75)

training_set <- training(data_split)
test_set  <- testing(data_split)

cv_set <- vfold_cv(training_set, strata = "lodgepole_pine", v = 10)

pine_recipe <- 
  recipe(
    lodgepole_pine ~ elevation + aspect + slope + horizontal_distance_to_hydrology +
      vertical_distance_to_hydrology + horizontal_distance_to_roadways + hillshade_9am + hillshade_noon + 
      hillshade_3pm + horizontal_distance_to_fire_points + wilderness_area + soil_type, 
    data = training_set
  ) %>%
  step_other(soil_type) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors())

lr_mod <- 
  logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")

lr_workflow <- 
  workflow() %>% 
  add_recipe(pine_recipe) %>%
  add_model(lr_mod)

lr_reg_grid <- tibble(penalty = 10^seq(-4, -1, length.out = 30))

lr_res <- 
  lr_workflow %>% 
  tune_grid(cv_set,
            grid = lr_reg_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))


lr_plot <- 
  lr_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

lr_plot 

top_models <-
  lr_res %>% 
  show_best(metric = "roc_auc", n = 15) %>% 
  arrange(penalty) 
top_models
