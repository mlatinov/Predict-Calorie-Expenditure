
#### Libraries ####
library(tidymodels)
library(baguette)
library(finetune)
library(tidyverse)
library(DALEX)
library(rules)
library(patchwork)

## Load the data 
data <- read_csv("data/train.csv")

# Bucket a numeric vector into n groups and sample 10% of each group 
sample_data <- data %>%
  mutate(
    strata = ntile(Calories,n = 50)
  ) %>%
  group_by(strata) %>%
  sample_frac(size = 0.1) %>%
  ungroup() %>%
  select(-strata)
  
# Split the data 
split <- initial_validation_split(data = sample_data)

# Train data
train_data <- training(split)

# Test data
test_data <- testing(split)

# Validation data
validation_data <- validation(split)

#### Preprocessing ####

## Baseline Preprocessing
recipe_ori <- recipe(Calories ~ .,data = train_data) %>%
  
  # Remove Id from the preprocessing and modeling
  update_role(id,new_role = "id") %>%
  
  # Remove near-zero var features
  step_nzv(all_nominal_predictors())%>%
  
  # Trasform all numeric features
  step_YeoJohnson(all_numeric_predictors()) %>%
  
  # Scale all Numerical Features
  step_scale(all_numeric_predictors()) %>%
  
  # Center all Numeric Features
  step_center(all_numeric_predictors()) %>%
  
  # Encode all categorical features
  step_dummy(all_nominal_predictors(),one_hot = TRUE)

## Feature Engineering
recipe_eng <- recipe(Calories ~ .,data = train_data) %>%
  
  # Remove Id from the preprocessing and modeling
  update_role(id,new_role = "id") %>%
  
  # Add new features
  step_mutate(
    
    # BMI Calculation
    bmi = Weight / (Height/100)^2,
    
    # The BMI prime calculation
    bmi_prime = bmi / 25,
    
    # Ponderal Index
    ponderal_index = Weight / (Height/100)^3,
    
    # Du Bois formula for BSA :
    bsa = 0.007184 * Weight^0.425 * Height^0.725,
    
    # The Keytel formula Burned By Heart Rate in Theory
    keytel_theoretical = case_when(
      Sex == "female" & Heart_Rate >= 40 & Heart_Rate <= 200 & Weight >= 30 & Weight <= 200 & Age >= 10 & Age <= 120 ~ 
        pmax((( -20.4022 + (0.4472 * Heart_Rate) - 
                  (0.1263 * Weight) + (0.074 * Age)) / 4.184) * Duration, 0),
      
      Sex == "male" & Heart_Rate >= 40 & Heart_Rate <= 200 & Weight >= 30 & Weight <= 200 & Age >= 10 & Age <= 120 ~ 
        pmax((( -55.0969 + (0.6309 * Heart_Rate) - 
                  (0.1988 * Weight) + (0.2017 * Age)) / 4.184) * Duration, 0),
      # Fallback case for invalid inputs
      TRUE ~ 0),
    
    # HRmax is estimated as
    hr_max = 208 - (0.7 * Age),
    
    # Heart Rate-based VO2max Estimation :
    vo2_max = 15 * (hr_max / 80),
    
    # Calories burned formulas based on a metabolic equation model using Vo2max:
    metabolic_vo2_cal = case_when(
      Sex == "male" ~
        ((-95.7735 + (0.634 * Heart_Rate) + (0.404 * vo2_max) + 
            (0.394 * Weight) + (0.271 * Age)) / 4.184) * Duration,
      Sex == "female" ~
        ((-59.3954 + (0.634 * Heart_Rate) + (0.380 * vo2_max ) + 
            (0.103 * Weight) + (0.274 * Age)) / 4.184) * Duration
    ),
    
    # The percentage of effort during exercise can be estimated using the heart rate reserve :
    effort = (Heart_Rate - 80) / (hr_max - 80),
    
    # To estimate the oxygen consumption at a given effort level using Vo2max
    vo2 = vo2_max * effort,
    
    # Calculate Metabolic Equivalent of Task
    met = vo2 / 3.5,
    
    # Calculate the total calories burned in a workout session
    calories_met_estimation = met * Weight * 0.0175 * Duration,
    
    # Change in core temperature (°C)
    delta_t = Body_Temp - 37,
    
    # Thermodynamic formula
    cal_thermodynamics = delta_t * Weight * 0.83,
    
    # Thermodynamic formula adjusted to heat losses
    cal_thermo_adjusted = cal_thermodynamics * 0.2,  
    
    # Calculate BMR (Mifflin-St Jeor Equation) with the formula
    bmr = case_when(
      Sex == "male" ~ 10 * Weight + 6.25 * Height - 5 * Age + 5,
      Sex == "female" ~ 10 * Weight + 6.25 * Height - 5 * Age - 161
    ),
    
    # Calculate BMR per min
    bmr_per_min = bmr / 1440,
    
    # Calc Burned Cal from bodyTemp evelation
    total_burn_from_body_heat = bmr_per_min * 2^((Body_Temp - 37)/10) * Duration,
    
    # Body Fat Percentage is calculated as
    bfp = case_when(
      Sex == "male" ~ 1.20 * bmi + 0.23 * Age - 16.2,
      Sex == "female" ~ 1.20 * bmi + 0.23 * Age - 5.4
    ),
    
    # Lean Mass (Fat-Free Mass) is calculated as
    lean_mass_kg = Weight * (1 - (bfp / 100)),
    
    # FFMI adjusted for height, is calculated as
    ffmi = lean_mass_kg / ((Height/100)^2) + 6.1 * (1.8 - (Height/100)),
    
    # Adjusted MET is calculated as
    met_ffmi_adjusted = met * (18 / ffmi),
    
    # Estimate calories burned from Adjusted MET with FFMI
    calories_met_ffmi_adjusted = met_ffmi_adjusted * Weight * (Duration / 60)
    
    ) %>%
  
  # Trasform all numeric features
  step_YeoJohnson(all_numeric_predictors()) %>%
  
  # Scale all Numerical Features
  step_scale(all_numeric_predictors()) %>%
  
  # Center all Numeric Features
  step_center(all_numeric_predictors()) %>%
  
  # Convert char into factors
  step_string2factor(all_nominal_predictors()) %>%
  
  # Encode all categorical features
  step_dummy(all_nominal_predictors())

#### Model Specifications ####

# Random Forest
ranger_model <- rand_forest(
  mtry = tune(), # Randomly Selected Predictors
  trees = 500,
  min_n = tune()
  )%>%
  set_mode("regression")%>%
  set_engine("ranger")

# XGB
xgb_model <- boost_tree(
  mtry = tune(), # Randomly Selected Predictors
  trees = tune(),
  min_n = tune(), # Minimal Node Size
  tree_depth = tune(),
  learn_rate = tune(),
  sample_size = tune(), # Proportion Observations Sampled
  loss_reduction = tune()
  )%>%
  set_mode("regression")%>%
  set_engine("xgboost")

# Elastic Net regression (L1 and L2 penalties)
elastic_net_model <- linear_reg(
  penalty = tune(),       # strength of regularization
  mixture = tune()        # 0 = ridge, 1 = lasso, in between = elastic net
  ) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

# Cubist rule-based regression models
cubist_model <- cubist_rules(
  committees = tune(), # Number of model ensembles
  neighbors = tune(),  # Instance-based correction
  max_rules = tune()   # The largest number of rules.
  )%>%
  set_mode("regression")%>%
  set_engine("Cubist")

# Bagged MARS 
bagged_mars <- bag_mars(
  num_terms = tune(),  # The number of features that will be retained in the final model
  prod_degree = tune() # Degree of Interaction
  )%>%
  set_mode("regression")%>%
  set_engine("earth")

# Bagged neural networks
bag_mlp <- bag_mlp(
  hidden_units = tune(),
  penalty = tune())%>%
  set_model_mode("regression")%>%
  set_engine("nnet")

## Create a workflow set
workfow_tuning_set <- workflow_set(
  preproc = list(engineered = recipe_eng,original = recipe_ori),
  models = list(
    bag_mars = bagged_mars,
    cubist_model = cubist_model,
    elastic_net = elastic_net_model,
    xgb_model = xgb_model,
    random_forest = ranger_model,
    bagged_nn = bag_mlp
  )
)
### Light tune for all the models with tune_race_anova

##  Create a custom metric RMSLE
rmsle_vec <- function(truth, estimate, na_rm = TRUE, ...) {
  rmsle_impl <- function(truth, estimate) {
    
    # Ensure values are positive 
    truth <- pmax(truth, 0)
    estimate <- pmax(estimate, 0)
    
    # Calc
    log_truth <- log1p(truth)
    log_estimate <- log1p(estimate)
    squared_errors <- (log_truth - log_estimate)^2
    mean_squared_error <- mean(squared_errors)
    sqrt(mean_squared_error)
  }
  # Template
  yardstick::metric_vec_template(
    metric_impl = rmsle_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    ...
  )
}

# Define RMSLE
rmsle <- function(data, truth, estimate, na_rm = TRUE, ...) {
  yardstick::metric_summarizer(
    metric_nm = "rmsle",
    metric_fn = rmsle_vec,
    data = data,
    truth = !!rlang::enquo(truth),
    estimate = !!rlang::enquo(estimate),
    na_rm = na_rm,
    ...
  )
}

# Register as yardstick metric
attr(rmsle, "direction") <- "minimize"
class(rmsle) <- c("numeric_metric", "metric", "function")

# Use in metric_set
custom_rmsle <- yardstick::metric_set(rmsle)

# Control Race 
control_anova <- control_race(
  randomize = TRUE,
  burn_in = 3, # The minimum number of resamples before eliminating the worst ones
  verbose = TRUE, 
  save_workflow = TRUE,
  save_pred = TRUE)

# Execute the Workflow_map tuning
workflow_map_light_tune <- workflow_map(
  object = workfow_tuning_set,
  fn = "tune_race_anova",
  grid = 20,
  metrics = custom_rmsle,
  resamples = vfold_cv(data = validation_data,v = 5),
  control = control_anova,
  verbose = TRUE,
  seed = 123)

# Collect the metrics
metrics <- collect_metrics(workflow_map_light_tune)

## Select best performance engineered models
best_bagged_mars <- workflow_map_light_tune %>% extract_workflow_set_result("engineered_bag_mars") %>%
  select_best()

best_cubist_model <- workflow_map_light_tune %>% extract_workflow_set_result("engineered_cubist_model") %>%
  select_best()

best_xgb_model <- workflow_map_light_tune %>% extract_workflow_set_result("engineered_xgb_model") %>%
  select_best()

best_random_forest <- workflow_map_light_tune %>% extract_workflow_set_result("engineered_random_forest") %>%
  select_best()

## Finalize the Models
bagged_mars_finalized <- bagged_mars %>% finalize_model(best_bagged_mars)
cubist_model_finalized <- cubist_model %>% finalize_model(best_cubist_model)
xgb_model_finalized <- xgb_model %>% finalize_model(best_xgb_model)
random_forest_finalized <- ranger_model %>% finalize_model(best_random_forest)

## Create Workflows
bagged_mars_workflow <- workflow()%>%
  add_model(bagged_mars_finalized) %>%
  add_recipe(recipe_eng)

cubist_model_workflow <- workflow() %>%
  add_model(cubist_model_finalized) %>%
  add_recipe(recipe_eng)

xgb_model_workflow <- workflow() %>%
  add_model(xgb_model_finalized) %>%
  add_recipe(recipe_eng)

random_forest_workflow <- workflow() %>%
  add_model(random_forest_finalized) %>%
  add_recipe(recipe_eng)

## Fit the models
bagged_mars_fit <- fit(bagged_mars_workflow,data = train_data)
cubist_model_fit <- fit(cubist_model_workflow,data = train_data)
xgb_model_fit <- fit(xgb_model_workflow,data = train_data)
random_forest_fit <- fit(random_forest_workflow,data = train_data)

# Preproc the data 
final_recipe <- prep(recipe_eng, training = train_data)
test_processed <- bake(final_recipe, new_data = test_data)
train_processed <- bake(final_recipe, new_data = train_data)

# Extract the the models
bagged_mars_final <- extract_fit_parsnip(bagged_mars_fit)
cubist_model_final <- extract_fit_parsnip(cubist_model_fit)
xgb_model_final <- extract_fit_parsnip(xgb_model_fit)
random_forest_final <- extract_fit_parsnip(random_forest_fit)

## Create Explainers 
mars_explainer <- DALEX::explain(
  model = bagged_mars_final,
  data = train_processed %>% select(-Calories),
  y = train_processed$Calories,
  label = "Bagged Mars"
)

cubist_explainer <- DALEX::explain(
  model = cubist_model_final,
  data = train_processed %>% select(-Calories),
  y = train_processed$Calories,
  label = "Cubist"
)

xgb_explainer <- DALEX::explain(
  model = xgb_model_final,
  data = train_processed %>% select(-Calories),
  y = train_processed$Calories,
  label = "XGB"
)

random_forest_explainer <- DALEX::explain(
  model = random_forest_final,
  data = train_processed %>% select(-Calories),
  y = train_processed$Calories,
  label = "Random Forest"
)

#### Champion–Challenger analysis ####

# Define DALEX custom loss function
rmsle_loss <- function(y_true, y_pred) {
  y_pred <- pmax(y_pred, 0)  
  sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
}

### Global Interpretability ###

## Residual diagnostics on the train data ##

## Random Forest

# Performance
random_forest_perf <- model_performance(random_forest_explainer)
rf_perf_p <- plot(random_forest_perf,geom = "histogram")

# Diagnostics
random_forest_diag <- model_diagnostics(random_forest_explainer)

# Y against residuals
rf_diag_y_resid <- plot(random_forest_diag, variable = "y", yvariable = "residuals")

# Y against y_hat
rf_diag_y_yhat <- plot(random_forest_diag, variable = "y", yvariable = "y_hat") + 
  geom_abline(colour = "red", intercept = 0, slope = 1)

### XGB

# Performance
xgb_model_perf <- model_performance(xgb_explainer)
xgb_perf_p <- plot(xgb_model_perf,geom = "histogram")

# Diagnostics
xgb_model_diag <- model_diagnostics(xgb_explainer)

# Y against residuals
xgb_model_diag_y_resid <- plot(xgb_model_diag,variable = "y",yvariable = "residuals") 

# Y against y_hat 
xgb_model_diag_y_yhat <- plot(xgb_model_diag,variable = "y",yvariable = "y_hat")+
  geom_abline(slope = 1,intercept = 0,colour = "red")

## MARS

# Performance
mars_model_perf <- model_performance(mars_explainer)
mars_perf_p <- plot(mars_model_perf,geom = "histogram")

# Diagnostics
mars_model_diag <- model_diagnostics(mars_explainer)

# Y against residuals
mars_model_diag_y_resid <- plot(mars_model_diag,variable = "y",yvariable = "residuals") 

# Y against y_hat
mars_model_diag_y_hat <- plot(mars_model_diag,variable = "y",yvariable = "y_hat")+
  geom_abline(slope = 1,intercept = 0,colour = "red")

## Cubist 

# Performance
cubist_model_perf <- model_performance(cubist_explainer)
cubist_model_perf_p <- plot(cubist_model_perf,geom = "histogram")

# Diagnostics
cubist_model_diag <- model_diagnostics(cubist_explainer)

# Y against residuals
cubist_model_diag_y_resid <- plot(cubist_model_diag,variable = "y",yvariable = "residuals") 

# Y against y_hat
cubist_model_diag_y_yhat <- plot(cubist_model_diag,variable = "y",yvariable = "y_hat")+
  geom_abline(slope = 1,intercept = 0,colour = "red")

# Plot all the result
histogram_residuals <- rf_perf_p + xgb_perf_p + mars_perf_p + cubist_model_perf_p
y_residuals <- cubist_model_diag_y_resid + mars_model_diag_y_resid + xgb_model_diag_y_resid + rf_diag_y_resid
y_yhat <- cubist_model_diag_y_yhat + mars_model_diag_y_hat + xgb_model_diag_y_yhat + rf_diag_y_yhat

## Permutation-based variable importance on the train_data

# Mars
mars_vip_50 <- model_parts(
  explainer = mars_explainer,
  type = "variable_importance",
  B = 50,
  loss_function = rmsle_loss
  )
# Plot the results 
plot(xgb_vip_50)

# Random Forest
random_forest_vip_50 <- model_parts(
  explainer = random_forest_explainer,
  type = "variable_importance",
  B = 50,
  loss_function = rmsle_loss
  )
# Plot the results 
plot(random_forest_vip_50)

# XGB
xgb_vip_50 <- model_parts(
  explainer = xgb_explainer,
  type = "variable_importance",
  B = 50,
  loss_function = rmsle_loss
)
# Plot the results 
plot(xgb_vip_50)

# Cubist
cubist_vip_50 <- model_parts(
  explainer = cubist_explainer,
  type = "variable_importance",
  B = 50,
  loss_function = rmsle_loss
)
# Plot the results 
plot(cubist_vip_50)

# Plot All the results
plot(random_forest_vip_50,xgb_vip_50,mars_vip_50,cubist_vip_50)+
  ggtitle("Mean variable-importance over 50 permutations", "") 

## ALE on the train data

# Random Forest 
rf_ale <- model_profile(
  explainer = random_forest_explainer,
  type = "accumulated",
  variables = c("bmi_prime","Height","ponderal_index","bmi","Weight"))

# Plot the results
rf_ale_plot <-plot(rf_ale)

# MARS
mars_ale <- model_profile(
  explainer = mars_explainer,
  type = "accumulated",
  variables = c("bmi_prime","Height","ponderal_index","bmi","Weight")
  )

# Plot the results
mars_ale_plot <- plot(mars_ale)

# XGB
xgb_ale <- model_profile(
  explainer = xgb_explainer,
  type = "accumulated",
  variables = c("bmi_prime","Height","ponderal_index","bmi","Weight")
  )

# Plot the results
xgb_ale_plot <- plot(xgb_ale)

# Cubist 
cubist_ale <- model_profile(
  explainer = cubist_explainer,type = "accumulated",
  variables = c("bmi_prime","Height","ponderal_index","bmi","Weight")
  )

# Plot the results
cubist_ale_plot <- plot(cubist_ale)

###  Local Interpretability ###

## Observations
low_cal <- train_processed[order(train_processed$Calories, decreasing = FALSE), ][1, , drop = FALSE]
high_cal <- train_processed[order(train_processed$Calories, decreasing = TRUE), ][1, , drop = FALSE]

## iBP

# Random Forest 

# High Calories
rf_ibp_high <- predict_parts(
  explainer = random_forest_explainer,
  new_observation = high_cal,
  type = "break_down_interactions")

# Low Calories
rf_ibp_low <- predict_parts(
  explainer = random_forest_explainer,
  new_observation = low_cal,
  type = "break_down_interactions")

# Plot the results
  
## LIME

## SHAP

## CP



