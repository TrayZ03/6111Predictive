library(tidymodels)
library(xgboost)
library(vip)

data <- read.csv("lab_data.csv", header = TRUE)
data$lodgepole_pine <- factor(data$lodgepole_pine)

# using the standard predictive analytics/machine learning approach with the tidymodels framework 
splits <- initial_split(data, strata = "lodgepole_pine", prop = 0.75)

training_set <- training(splits)
test_set  <- testing(splits)

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

xgboost_model <- boost_tree(
      trees = tune(),
      tree_depth = tune(),
      learn_rate = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgboost_workflow <- 
  workflow() %>% 
  add_recipe(pine_recipe) %>%
  add_model(xgboost_model)

################### use tune_grid to find best hyperparameter values #####################

xgboost_grid <- grid_latin_hypercube(
  trees(),
  tree_depth(),
  learn_rate(),
  size = 10
)

xgboost_results <- xgboost_workflow %>%
  tune_grid(resamples = cv_set,
            grid = xgboost_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

xgboost_results %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, trees:learn_rate) %>%
  pivot_longer(trees:learn_rate,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")

xgboost_results %>% 
  show_best(metric = "roc_auc")

########################## use best hyperparameter values ################################

best_auc <- xgboost_results %>% select_best("roc_auc")

final_xgboost_workflow <- finalize_workflow(
  xgboost_workflow,
  best_auc
)

last_xgboost_fit <- 
  final_xgboost_workflow %>% 
  last_fit(splits)

################################### feature importance ###################################

last_xgboost_fit %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 20)

##################################### plot ROC curve #####################################

test_results <-
  test_set %>%
  select(lodgepole_pine) %>%
  bind_cols(
    last_xgboost_fit %>% collect_predictions() %>%
      select(p_1 = .pred_1)
  )

roc_data <- data.frame(threshold=seq(1,0,-0.01), fpr=0, tpr=0)
for (i in roc_data$threshold) {
  
  over_threshold <- test_results[test_results$p_1 >= i, ]
  
  fpr <- sum(over_threshold$lodgepole_pine==0)/sum(test_results$lodgepole_pine==0)
  roc_data[roc_data$threshold==i, "fpr"] <- fpr
  
  tpr <- sum(over_threshold$lodgepole_pine==1)/sum(test_results$lodgepole_pine==1)
  roc_data[roc_data$threshold==i, "tpr"] <- tpr
  
}

ggplot() +
  geom_line(data = roc_data, aes(x = fpr, y = tpr, color = threshold), linewidth = 2) +
  scale_color_gradientn(colors = rainbow(3)) +
  geom_abline(intercept = 0, slope = 1, lty = 2) +
  geom_point(data = roc_data[seq(1, 101, 10), ], aes(x = fpr, y = tpr)) +
  geom_text(data = roc_data[seq(1, 101, 10), ],
            aes(x = fpr, y = tpr, label = threshold, hjust = 1.2, vjust = -0.2))


############################# ROC curve calculation breakdown ############################

ggplot(data = test_results, aes(x = p_1, y = lodgepole_pine)) +
  geom_jitter()

threshold <- 0.6

test_results$predictions <- ifelse(test_results$p_1 >= threshold, 1, 0)
tp <- nrow(test_results[test_results$lodgepole_pine==1 & test_results$predictions==1, ])
fp <- nrow(test_results[test_results$lodgepole_pine==0 & test_results$predictions==1, ])
tn <- nrow(test_results[test_results$lodgepole_pine==0 & test_results$predictions==0, ])
fn <- nrow(test_results[test_results$lodgepole_pine==1 & test_results$predictions==0, ])

test_results$type <- ""
test_results[test_results$lodgepole_pine==1 & test_results$predictions==1, "type"] <- "tp"
test_results[test_results$lodgepole_pine==0 & test_results$predictions==1, "type"] <- "fp"
test_results[test_results$lodgepole_pine==0 & test_results$predictions==0, "type"] <- "tn"
test_results[test_results$lodgepole_pine==1 & test_results$predictions==0, "type"] <- "fn"

ggplot(data = test_results, aes(x = p_1, y = lodgepole_pine)) +
  geom_jitter(aes(colour = type)) +
  geom_vline(xintercept = threshold, linetype = "dashed", color = "blue", linewidth = 1.5) +
  scale_color_brewer(palette = "RdYlBu")

fpr <- fp/(fp + tn)
tpr <- tp/(tp + fn)


