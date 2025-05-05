
#### Libraries ####
library(tidymodels)
library(baguette)
library(tidyverse)
library(rules)

## Load the data 
data <- read_csv("data/train.csv")

# Split the data 
split <- initial_validation_split(data = data)

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
  add_role(id,new_role = "id") %>%
  
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
  add_role(id,new_role = "id") %>%
  
  # Add new features
  step_mutate(
    
    # BMI Calculation
    bmi = Weight / (Height/100)^2,
    
    # Classification BMI
    classification_BMI = case_when(
      bmi < 25 ~ "Normal",
      bmi >= 25 ~ "Overweight"
    ),
    
    # The BMI prime calculation
    bmi_prime = bmi / 25,
    
    # Classification BMI Prime
    classification_BMI_Prime = case_when(
      bmi_prime < 1 ~ "Normal_prime",
      bmi_prime >= 1 ~ "Overweight_1_prime"
    ),
    
    # Ponderal Index
    ponderal_index = Weight / (Height/100)^3,
    
    # Classification PI
    classification_PI = case_when(
      ponderal_index < 15 ~ "Normal_pi",
      ponderal_index >= 15 ~ "Overweight_1_pi"
    ),
    
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
    calories_met_ffmi_adjusted = met_ffmi_adjusted * Weight * (Duration / 60) %>%
  
  # Trasform all numeric features
  step_YeoJohnson(all_numeric_predictors()) %>%
  
  # Scale all Numerical Features
  step_scale(all_numeric_predictors()) %>%
  
  # Center all Numeric Features
  step_center(all_numeric_predictors()) %>%
  
  # Remove highly correlated features
  step_corr(all_numeric_predictors(), threshold = 0.9,trained = TRUE) %>%
  
  # Encode all categorical features
  step_dummy(all_nominal_predictors(),one_hot = TRUE))

#### Model Specifications ####

# Random Forest
ranger_model <- rand_forest(
  mtry = tune(), # Randomly Selected Predictors
  trees = 500,
  min_n = tune())%>%
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
linear_reg <- linear_reg(mixture = tune(),penalty = tune())%>%
  set_mode("regression")%>%
  set_engine("lm")

# Generalized additive model
gam_model <- gen_additive_mod(
  select_features = TRUE,
  adjust_deg_free = tune() # Smoothness Adjustment 
  )%>%
  set_mode("regression")%>%
  set_engine("mgcv")

# Cubist rule-based regression models
cubist_model <- cubist_rules(
  committees = tune(), # Number of model ensembles
  neighbors = tune(), # Instance-based correction
  max_rules = tune() # The largest number of rules.
  )%>%
  set_mode("regression")%>%
  set_engine("Cubist")

# Bagged MARS 
bagged_mars <- bag_mars(
  num_terms = tune(), # The number of features that will be retained in the final model
  prod_degree = tune() # Degree of Interaction
  )%>%
  set_mode("regression")%>%
  set_engine("earth")

## Create a workflow set
workfow_tuning_set <- workflow_set(
  preproc = list(original = recipe_ori,engineered = recipe_eng),
  models = list(
    bag_mars = bagged_mars,
    cubist_model = cubist_model,
    Gam = gam_model,
    elastic_net = linear_reg,
    xgb_model = xgb_model,
    random_forest = ranger_model
  )
)
## Light tune for all the models with tune_race_anova
workflow_map_light_tune <- workflow_map(
  object = workfow_tuning_set,
  fn = "tune_race_anova",
  grid = 20,
  resamples = vfold_cv(data = validation_data,v = 5),
  verbose = TRUE,
  seed = 123)



