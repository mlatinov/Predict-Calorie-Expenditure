
#### Libraries ####
library(tidymodels)
library(baguette)
library(finetune)
library(tidyverse)
library(DALEX)
library(rules)
library(patchwork)
library(DiceDesign)

tidymodels_prefer()

## Load the data 
data <- read_csv("data/train.csv")

# Bucket a numeric vector into n groups and sample 30% of each group 
sample_data <- data %>%
  mutate(
    strata = ntile(Calories,n = 50)
  ) %>%
  group_by(strata) %>%
  sample_frac(size = 0.3) %>%
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
  step_dummy(all_nominal_predictors())

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
    ffmi = lean_mass_kg / ((Height/100)^2) + 6.1 * pmax(0, 1.8 - (Height/100)),
    
    # Adjusted MET is calculated as
    met_ffmi_adjusted = met * (1 + 18 / ffmi),
    
    # Estimate calories burned from Adjusted MET with FFMI
    calories_met_ffmi_adjusted = met_ffmi_adjusted * Weight * (Duration / 60)
    
    ) %>%
  
  # Transform all numeric features
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

# Cubist rule-based regression models
cubist_model <- cubist_rules(
  committees = tune(), # Number of model ensembles
  neighbors = tune(),  # Instance-based correction
  max_rules = tune()   # The largest number of rules.
  )%>%
  set_mode("regression")%>%
  set_engine("Cubist")

## Create a workflow set
workfow_tuning_set <- workflow_set(
  preproc = list(engineered = recipe_eng,original = recipe_ori),
  models = list(
    cubist_model = cubist_model,
    xgb_model = xgb_model,
    random_forest = ranger_model
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

# Plot the result
metrics %>%
  mutate(recipe_type = ifelse(str_detect(wflow_id, "^engineered"), "eng", "ori")) %>%
  group_by(wflow_id, recipe_type) %>%
  summarise(mean = mean(mean), .groups = "drop") %>%
  ggplot(aes(x = mean, y = fct_reorder(as.factor(wflow_id),.x = mean,.fun = sum,.desc = TRUE), fill = recipe_type)) +
  geom_col() +
  geom_text(aes(label = round(mean, 4)), hjust = -0.1) + 
  theme_minimal()

#### Define a param space ####

## Select top performance engineered models
best_cubist_model <- workflow_map_light_tune %>% extract_workflow_set_result("engineered_cubist_model") %>%
  show_best(n = 1) %>%
  select(-.metric, -.estimator, -mean, -n, -std_err, -.config)

best_xgb_model <- workflow_map_light_tune %>% extract_workflow_set_result("engineered_xgb_model") %>%
  show_best(n = 1) %>%
  select(-.metric, -.estimator, -mean, -n, -std_err, -.config)

best_random_forest <- workflow_map_light_tune %>% extract_workflow_set_result("engineered_random_forest") %>%
  show_best(n = 1) %>%
  select(-.metric, -.estimator, -mean, -n, -std_err, -.config)

# Cubist

# Initial param space
cubist_param_space <- parameters(
  committees(range = range(best_cubist_model$committees) + c(-5, 5)),
  neighbors(range = c(0, 9)),
  max_rules(range = range(best_cubist_model$max_rules) + c(-20, 20))
)

# Get the numeric ranges for each parameter
param_ranges_cubist <- purrr::map(cubist_param_space$object, function(x) {
  rng <- dials::range_get(x)
  as.numeric(rng)
})

# Convert the list to a data frame with sequences
param_df_cubist <- data.frame(
  committees = as.integer(seq(param_ranges_cubist[[1]][1], param_ranges_cubist[[1]][2], length.out = 10)),
  neighbors = as.integer(seq(param_ranges_cubist[[2]][1], param_ranges_cubist[[2]][2], length.out = 10)),
  max_rules = seq(param_ranges_cubist[[3]][1], param_ranges_cubist[[3]][2], length.out = 10)
)

# Normalize the space (rescale between 0 and 1)
norm_space_cubist <- param_df_cubist %>%
  mutate(across(everything(), rescale))

# Check for Irregular Coverage (Heuristics)
dist_matrix <- dist(as.matrix(param_df_cubist))
cv_dist<- sd(dist_matrix) / mean(dist_matrix) # 0.6098367

# LCH design
lch_cubust_design <- lhsDesign(n = 20, dimension = 3, randomized = TRUE,seed = 123)$design

# Simulated annealing (SA) routine for Latin Hypercube Sample (LHS) optimization 
optimized_lch_cubist <- maximinSA_LHS(
  design = lch_cubust_design,
  T0=10,                      # The initial temperature of the SA algorithm
  c=0.99,                     # A constant parameter regulating how the temperature goes down
  it=1000,                    # The number of iterations
  profile="GEOM_MORRIS")

# Map the design into the params
cubist_final_design <- data.frame(
  committees = param_df_cubist$committees[cut(optimized_lch_cubist$design[,1], 
                                       breaks = seq(0,1,length=11), 
                                       labels = FALSE)],
  
  neighbors = param_df_cubist$neighbors[cut(optimized_lch_cubist$design[,2], 
                                     breaks = seq(0,1,length=11), 
                                     labels = FALSE)],
  
  max_rules = param_df_cubist$max_rules[cut(optimized_lch_cubist$design[,3], 
                                     breaks = seq(0,1,length=11), 
                                     labels = FALSE)]
)

# XGB 

# Initial adaptive ranges from the best configurations
xgb_initial_param_space <- parameters(
  
  # Feature selection
  mtry(range = c(
    max(1, floor(best_xgb_model$mtry* 0.5)),   # 50% below
    min(best_xgb_model$mtry * 2, 15)           # 100% above, max 15
  )),
  
  # Tree structure
  trees(range = c(
    max(50, floor(best_xgb_model$trees * 0.7)), # 30% below
    ceiling(best_xgb_model$trees * 1.5)         # 50% above
  )),
  
  min_n(range = c(
    max(2, floor(best_xgb_model$min_n * 0.5)),  # 50% below
    ceiling(best_xgb_model$min_n * 1.5)         # 50% above
  )),
  
  tree_depth(range = c(
    max(2, best_xgb_model$tree_depth - 3),      # Absolute -3
    best_xgb_model$tree_depth + 3               # Absolute +3
  )),
  
  # Learning dynamics (log scales)
  learn_rate(range = c(
    log10(best_xgb_model$learn_rate) - 1,       # 1 order magnitude below
    log10(min(0.1, best_xgb_model$learn_rate * 10)) # 1 order above
  ), trans = log10_trans()),
  
  loss_reduction(range = c(
    log10(best_xgb_model$loss_reduction) - 1,   # 1 order below
    log10(best_xgb_model$loss_reduction) + 2    # 2 orders above
  ), trans = log10_trans()),
  
  # Sampling
  sample_prop(range = c(
    max(0.1, best_xgb_model$sample_size), 
    min(0.9, best_xgb_model$sample_size + 0.2)  # +0.2 absolute
  ))
)
  
# Get the numeric ranges for each parameter
xgb_param_ranges <- purrr::map(xgb_initial_param_space$object,function(x){
  rng <- range_get(x)
  as.numeric(rng)
})

# Convert the list to a data frame with sequences
param_df_xgb <- data.frame(
  mtry = seq(xgb_param_ranges[[1]][1], xgb_param_ranges[[1]][2], length.out = 10),
  trees = as.integer(seq(xgb_param_ranges[[2]][1], xgb_param_ranges[[2]][2], length.out = 10)),
  min_n = as.integer(seq(xgb_param_ranges[[3]][1], xgb_param_ranges[[3]][2], length.out = 10)),
  tree_depth =  as.integer(seq(xgb_param_ranges[[4]][1], xgb_param_ranges[[4]][2], length.out = 10)),
  learn_rate = 10^seq(xgb_param_ranges[[5]][1], xgb_param_ranges[[5]][2], length.out = 10),
  loss_reduction = 10^seq(xgb_param_ranges[[6]][1], xgb_param_ranges[[6]][2], length.out = 10),
  sample_prop = seq(xgb_param_ranges[[7]][1], xgb_param_ranges[[7]][2], length.out = 10)
)

# Create normalized parameter space
norm_space_xgb <- param_df_xgb %>%
  mutate(across(everything(),rescale))

# Check for Irregular Coverage (Heuristics)
dist_matrix <- dist(as.matrix(norm_space_xgb))
cv_dist_xgb <- sd(dist_matrix) / mean(dist_matrix)

#  Generate initial LHS design 
lhc_xgb_initial <- lhsDesign(n = 50,dimension = 7,randomized = TRUE,seed = 123)$design

# Run max_min optimization 
optimized_lch_xgb <- maximinSA_LHS(
  design =lhc_xgb_initial,
  T0=20,                      # The initial temperature of the SA algorithm
  c=0.95,                     # A constant parameter regulating how the temperature goes down
  it=2000,                    # The number of iterations
  profile="GEOM_MORRIS"
)
  
# Map the optimized max_min LHC 
xgb_final_design <- data.frame(
  mtry = param_df_xgb$mtry[findInterval(optimized_lch_xgb$design[,1], 
                                        seq(0,1,length.out=11))],
  
  trees = param_df_xgb$trees[findInterval(optimized_lch_xgb$design[,2],
                                          seq(0,1,length.out=11))],
  
  min_n = param_df_xgb$min_n[findInterval(optimized_lch_xgb$design[,3],
                                          seq(0,1,length.out=11))],
  
  tree_depth = param_df_xgb$tree_depth[findInterval(optimized_lch_xgb$design[,4],
                                                    seq(0,1,length.out=11))],
  
  learn_rate = param_df_xgb$learn_rate[findInterval(optimized_lch_xgb$design[,5],
                                                    seq(0,1,length.out=11))],
  
  loss_reduction = param_df_xgb$loss_reduction[findInterval(optimized_lch_xgb$design[,6],
                                                            seq(0,1,length.out=11))],
  
  sample_prop = param_df_xgb$sample_prop[findInterval(optimized_lch_xgb$design[,7],
                                                      seq(0,1,length.out=11))]
)

## Check for Irregular Coverage (Heuristics)
norm_space_xgb <- xgb_final_design %>%
  mutate(across(everything(),rescale))
dist_matrix <- dist(as.matrix(norm_space_xgb))
cv_dist_xgb <- sd(dist_matrix) / mean(dist_matrix) # 0.2118277

## Random Forest 

# Initial adaptive ranges from the best configurations
rf_param_space <- parameters(
  mtry(range = c(
    max(1, floor(min(best_random_forest$mtry) * 0.7)),  # 30% below lowest best
    ceiling(max(best_random_forest$mtry) * 1.3)         # 30% above highest best
  )),
  
  min_n(range = c(
    max(1, floor(min(best_random_forest$min_n) * 0.5)), # 50% below lowest best
    ceiling(max(best_random_forest$min_n) * 1.5)         # 50% above highest best
  ))
)

# Get numeric ranges for LHS sampling
rf_ranges <- map(rf_param_space$object, ~ as.numeric(range_get(.x)))

# Convert the list to a data frame with sequences
param_df_rf <- data.frame(
  mtry = seq(rf_ranges[[1]][1], rf_ranges[[1]][2], length.out = 10),
  min_n = seq(rf_ranges[[2]][1], rf_ranges[[2]][2], length.out = 10)
)

## Check for Irregular Coverage (Heuristics)
norm_space_rf <- param_df_rf %>%
  mutate(across(everything(),rescale))
dist_matrix <- dist(as.matrix(norm_space_rf))
cv_dist_rf <- sd(dist_matrix) / mean(dist_matrix) # 0.6098367

#  Generate initial LHS design 
lhc_rf_initial <- lhsDesign(n = 20,dimension = 2,randomized = TRUE,seed = 123)$design

# Run maximin optimization 
optimized_lch_rf <- maximinSA_LHS(
  design =lhc_rf_initial,
  T0=10,                      # The initial temperature of the SA algorithm
  c=0.97,                     # A constant parameter regulating how the temperature goes down
  it=2000,                    # The number of iterations
  profile="GEOM_MORRIS")

# Map the design into the params
rf_final_design <- data.frame(
  mtry = param_df_xgb$mtry[findInterval(optimized_lch_rf$design[,1], 
                                        seq(0,1,length.out=11))],
  
  min_n = param_df_xgb$min_n[findInterval(optimized_lch_rf$design[,2],
                                          seq(0,1,length.out=11))]
)

## Check for Irregular Coverage (Heuristics)
norm_space_rf <- rf_final_design %>%
  mutate(across(everything(),rescale))
dist_matrix <- dist(as.matrix(norm_space_rf))
cv_dist_rf <- sd(dist_matrix) / mean(dist_matrix)  # 0.456328


#### MBO ####

## Create Workflows
cubist_model_workflow <- workflow() %>%
  add_model(cubist_model) %>%
  add_recipe(recipe_eng)

xgb_model_workflow <- workflow() %>%
  add_model(xgb_model) %>%
  add_recipe(recipe_eng)

random_forest_workflow <- workflow() %>%
  add_model(ranger_model) %>%
  add_recipe(recipe_eng)


# Initial with tune grid()
rf_initial <- tune_grid(
  object = random_forest_workflow,
  resamples = vfold_cv(data = validation_data,v = 5),
  grid = rf_final_design,
  metrics =custom_rmsle,
  control = control_grid(verbose = TRUE,save_pred = TRUE)
)

xgb_initial <- tune_grid(
  object = xgb_model_workflow,
  resamples = vfold_cv(data = validation_data,v = 5),
  grid = xgb_final_design,
  metrics = custom_rmsle,
  control = control_grid(verbose = TRUE,save_pred = TRUE)
)

cubist_initial <- tune_grid(
  object = cubist_model_workflow,
  resamples = vfold_cv(data = validation_data,v = 5),
  grid = cubist_final_design,
  metrics = custom_rmsle,
  control = control_grid(verbose = TRUE,save_pred = TRUE)
)

## Set Up Bayesian Control Parameters
bayes_control <- control_bayes(
  verbose = TRUE,
  verbose_iter = TRUE,
  no_improve = 15,
  save_pred = TRUE,
  save_workflow = TRUE,
  seed = 123
)

# Resamlpes 
resamples <- vfold_cv(data = validation_data,v = 5)

# Param_info 
rf_params_info <- parameters(
  mtry(range = range(rf_final_design$mtry)),
  min_n(range = range(rf_final_design$min_n))
)

xgb_param_info <- parameters(
  mtry(range = range(xgb_final_design$mtry)),
  min_n(range = range(xgb_final_design$min_n)),
  tree_depth(range = range(xgb_final_design$tree_depth)),
  learn_rate(range = range(xgb_final_design$learn_rate)),
  loss_reduction(range = range(xgb_final_design$loss_reduction)),
  sample_size(range = range(xgb_final_design$sample_size))
)

cubist_param_info <- parameters(
  committees(range = range(cubist_final_design$committees)),
  neighbors(range = range(cubist_final_design$neighbors)),
  max_rules(range = range(cubist_final_design$max_rules))
)

## Run tune_bayes()

bayes_rf <- tune_bayes(
  object = random_forest_workflow,
  resamples = resamples,
  iter = 30,
  param_info = rf_params_info,
  initial = rf_initial,
  metrics = custom_rmsle,
  control = bayes_control
)

bayes_xgb <- tune_bayes(
  object = xgb_model_workflow,
  resamples = resamples,
  iter = 50,
  param_info = xgb_param_info,
  initial = xgb_initial,
  metrics = custom_rmsle,
  control = bayes_control
)

bayes_cubist <- tune_bayes(
  object = cubist_model_workflow,
  resamples = resamples,
  iter = 30,
  param_info = cubist_param_info,
  initial = cubist_initial,
  metrics = custom_rmsle,
  control = bayes_control
)
  
## Viz the results

## Select best params from MBO


## Finalize the workflows


## Fit the models
cubist_model_fit <- fit(cubist_model_workflow,data = train_data)
xgb_model_fit <- fit(xgb_model_workflow,data = train_data)
random_forest_fit <- fit(random_forest_workflow,data = train_data)

## Predict on the test data 


#### Champion–Challenger analysis ####

# Preproc the data 
final_recipe <- prep(recipe_eng, training = train_data)
test_processed <- bake(final_recipe, new_data = test_data)
train_processed <- bake(final_recipe, new_data = train_data)

# Extract the the models
cubist_model_final <- extract_fit_parsnip(cubist_model_fit)
xgb_model_final <- extract_fit_parsnip(xgb_model_fit)
random_forest_final <- extract_fit_parsnip(random_forest_fit)

## Create Explainers 

# Cubist explainer
cubist_explainer <- DALEX::explain(
  model = cubist_model_final,
  data = train_processed %>% select(-Calories),
  y = train_processed$Calories,
  label = "Cubist"
)

# XGB explainer
xgb_explainer <- DALEX::explain(
  model = xgb_model_final,
  data = train_processed %>% select(-Calories),
  y = train_processed$Calories,
  label = "XGB"
)

# Random Forest_explainer
random_forest_explainer <- DALEX::explain(
  model = random_forest_final,
  data = train_processed %>% select(-Calories),
  y = train_processed$Calories,
  label = "Random Forest"
)

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
histogram_residuals <- rf_perf_p + xgb_perf_p  + cubist_model_perf_p
y_residuals <- cubist_model_diag_y_resid  + xgb_model_diag_y_resid + rf_diag_y_resid
y_yhat <- cubist_model_diag_y_yhat  + xgb_model_diag_y_yhat + rf_diag_y_yhat

## Permutation-based variable importance on the train_data

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
plot(random_forest_vip_50,xgb_vip_50,cubist_vip_50)+
  ggtitle("Mean variable-importance over 50 permutations", "") 

## ALE on the train data

# Random Forest 
rf_ale <- model_profile(
  explainer = random_forest_explainer,
  type = "accumulated",
  variables = c("bmi_prime","Height","ponderal_index","bmi","Weight"))

# Plot the results
rf_ale_plot <-plot(rf_ale)

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

rf_ibp_high_p <- plot(rf_ibp_high)

# Low Calories
rf_ibp_low <- predict_parts(
  explainer = random_forest_explainer,
  new_observation = low_cal,
  type = "break_down_interactions")

rf_ibp_low_p <- plot(rf_ibp_low)

##### Finalize the Models ####

## Fit the models on the entire data 

## Predict on the test_set

## Write a csv for  kaggle







