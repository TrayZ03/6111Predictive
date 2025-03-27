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


################################### ROC curve calculation breakdown ###################################

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

################################### hypothetical example ###################################
# consider the following situation: for some threshold, we have a classifier that can perfectly
# separate the actual positives and actual negatives in the test set

test_results_hypothetical <- test_results %>% select(lodgepole_pine, p_1)
test_results_hypothetical[test_results_hypothetical$lodgepole_pine==1, "p_1"] <- 
  runif(nrow(test_results_hypothetical[test_results_hypothetical$lodgepole_pine==1, ]), min = 0.751, max = 1)

test_results_hypothetical[test_results_hypothetical$lodgepole_pine==0, "p_1"] <- 
  runif(nrow(test_results_hypothetical[test_results_hypothetical$lodgepole_pine==0, ]), min = 0, max = 0.749)

############################################################################################

ggplot(data = test_results_hypothetical, aes(x = p_1, y = lodgepole_pine)) +
  geom_jitter()

threshold <- 0.6

test_results_hypothetical$predictions <- ifelse(test_results_hypothetical$p_1 >= threshold, 1, 0)
tp <- nrow(test_results_hypothetical[test_results_hypothetical$lodgepole_pine==1 & test_results_hypothetical$predictions==1, ])
fp <- nrow(test_results_hypothetical[test_results_hypothetical$lodgepole_pine==0 & test_results_hypothetical$predictions==1, ])
tn <- nrow(test_results_hypothetical[test_results_hypothetical$lodgepole_pine==0 & test_results_hypothetical$predictions==0, ])
fn <- nrow(test_results_hypothetical[test_results_hypothetical$lodgepole_pine==1 & test_results_hypothetical$predictions==0, ])

test_results_hypothetical$type <- ""
test_results_hypothetical[test_results_hypothetical$lodgepole_pine==1 & test_results_hypothetical$predictions==1, "type"] <- "tp"
test_results_hypothetical[test_results_hypothetical$lodgepole_pine==0 & test_results_hypothetical$predictions==1, "type"] <- "fp"
test_results_hypothetical[test_results_hypothetical$lodgepole_pine==0 & test_results_hypothetical$predictions==0, "type"] <- "tn"
test_results_hypothetical[test_results_hypothetical$lodgepole_pine==1 & test_results_hypothetical$predictions==0, "type"] <- "fn"

ggplot(data = test_results_hypothetical, aes(x = p_1, y = lodgepole_pine)) +
  geom_jitter(aes(colour = type)) +
  geom_vline(xintercept = threshold, linetype = "dashed", color = "blue", linewidth = 1.5) +
  scale_color_brewer(palette = "RdYlBu")

fpr <- fp/(fp + tn)
tpr <- tp/(tp + fn)

################################### plot ROC curve ###################################

roc_data <- data.frame(threshold=seq(1,0,-0.01), fpr=0, tpr=0)
for (i in roc_data$threshold) {
  
  over_threshold <- test_results_hypothetical[test_results_hypothetical$p_1 >= i, ]
  
  fpr <- sum(over_threshold$lodgepole_pine==0)/sum(test_results_hypothetical$lodgepole_pine==0)
  roc_data[roc_data$threshold==i, "fpr"] <- fpr
  
  tpr <- sum(over_threshold$lodgepole_pine==1)/sum(test_results_hypothetical$lodgepole_pine==1)
  roc_data[roc_data$threshold==i, "tpr"] <- tpr
  
}

ggplot() +
  geom_line(data = roc_data, aes(x = fpr, y = tpr, color = threshold), linewidth = 2) +
  scale_color_gradientn(colors = rainbow(3)) +
  geom_abline(intercept = 0, slope = 1, lty = 2) +
  geom_point(data = roc_data[seq(1, 101, 10), ], aes(x = fpr, y = tpr)) +
  geom_text(data = roc_data[seq(1, 101, 10), ],
            aes(x = fpr, y = tpr, label = threshold, hjust = 1.2, vjust = -0.2))
