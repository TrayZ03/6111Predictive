library(tidymodels)

data <- read.csv("lab_data.csv", header = TRUE)
data$lodgepole_pine <- factor(data$lodgepole_pine)

# using the standard predictive analytics/machine learning approach with the tidymodels framework 
data_split <- initial_split(data, strata = "lodgepole_pine", prop = 0.75)

training_set <- training(data_split)
test_set  <- testing(data_split)

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
  step_scale(all_predictors()) %>%
  prep(training = training_set)

logreg_fit <- 
  logistic_reg() %>%
  set_engine("glm") %>%
  fit(lodgepole_pine ~ ., data = bake(pine_recipe, new_data = training_set))
logreg_fit

test_baked <- bake(pine_recipe, new_data = test_set, all_predictors())

test_results <- 
  test_set %>%
  select(lodgepole_pine) %>%
  bind_cols(
    predict(logreg_fit, new_data = test_baked, type = "prob") %>%
      select(p_1 = .pred_1)
  )

################################### plot ROC curve ###################################

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

################################### plot calibration curve ###################################

calibration_data <- data.frame(bin_midpoint=seq(0.05,0.95,0.1),
                               observed_event_percentage=0)
for (i in seq(0.05,0.95,0.1)) {
  
  in_interval <- test_results[test_results$p_1 >= (i-0.05) & test_results$p_1 <= (i+0.05), ]
  oep <- nrow(in_interval[in_interval$lodgepole_pine==1, ])/nrow(in_interval)
  calibration_data[calibration_data$bin_midpoint==i, "observed_event_percentage"] <- oep
  
}

ggplot(data = calibration_data, aes(x = bin_midpoint, y = observed_event_percentage)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, lty = 2) +
  geom_point(size = 2) +
  geom_text(aes(label = bin_midpoint), hjust = 0.75, vjust = -0.5)
