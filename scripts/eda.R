
#### Libraries ####
library(tidyverse)
library(ranger)
library(mgcv)
library(ggcorrplot)
library(ggfortify)
library(viridis)
library(plotly)

## Load the data 
data_train <- read_csv("data/train.csv",n_max = 50000)

## Mutate  New  features
data_train <- data_train %>%
  mutate(
    
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
    
    # **HRmax** is estimated as
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
  
  # Change Char into Factors
  mutate(across(where(is.character), as.factor))
    
#### Univariate Analysis ####

# Numerical Features 
data_train %>%
  select_if(is.numeric)%>%
  select(-id) %>%
  pivot_longer(cols = everything(),names_to = "Feature",values_to = "Value")%>%
  ggplot(aes(x = Value,y = Feature)) +
  geom_boxplot() +
  geom_violin(alpha = 0.2,fill = "lightblue") +
  facet_wrap(~Feature,scale = "free")+
  theme_minimal()+
  labs(
    title = "VB Plot" 
  )+
  theme(
    title = element_text(size = 20,face = "bold"),
    axis.text.y.left = element_blank())

data_train %>%
  select_if(is.numeric)%>%
  select(-id) %>%
  pivot_longer(cols = everything(),names_to = "Feature",values_to = "Value")%>%
  ggplot(aes(x = Value))+
  geom_histogram(aes(y = after_stat(density)),fill = "grey80", color = "black", bins = 50)+
  geom_density(colour = "red",alpha = 0.5)+
  facet_wrap(~Feature,scale = "free")+
  theme_classic()+
  labs(
    title = "Numerical Distributions",
    x = "",
    y = ""
    )+
theme(
  title = element_text(size = 20 , face = "bold")
)

## Categorical Features 
data_train %>%
  select(where(is.factor)) %>%
  pivot_longer(cols = everything(), names_to = "Feature", values_to = "Value") %>%
  group_by(Feature, Value) %>%
  summarise(Count = n(), .groups = "drop") %>%
  mutate(Value = fct_reorder(Value, Count)) %>%
  ggplot(aes(x = Value, y = Count, fill = Feature)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  geom_text(aes(label = Count), hjust = -0.1, size = 3) +
  facet_wrap(~Feature, scales = "free") +
  coord_flip() +
  theme_minimal(base_size = 13) +
  theme(
    strip.text = element_text(face = "bold"),
    axis.text.y = element_text(size = 9),
    axis.text.x = element_text(size = 9)
  ) +
  labs(x = NULL, y = "Count", title = "Categorical Feature Distributions")

#### Bivariate Analysis ####

# Loss Function
rmsle <- function(actual, predicted) {
  sqrt(mean((log1p(predicted) - log1p(actual))^2))
}

# Bivariate target ~ Numerical 
bivariate_numerical <- function(data,x,y){
  
  # Correlation
  cor_test <- cor.test(data[[x]],data[[y]])
  cor <- cor(data[[x]],data[[y]])
  
  # Linear Regression
  lm <- lm(formula = data[[y]] ~ data[[x]],data = data)
  summary_lm <- summary(lm)
  
  # GAM 
  gam <- gam(formula = as.formula(paste(y, "~ s(", x, ")")), data = data)
  summary_gam <- summary(gam)
  
  # Scatter Plot
  plot <- ggplot(data = data,aes(x = data[[x]],y = data[[y]]))+
    geom_point()+
    geom_smooth(method = "gam",colour = "blue")+
    geom_smooth(method = "lm",colour = "red")+
    theme_minimal()+
    labs(
      title = paste0("Scatter plot between ",y, " and ",x,""),
      x = x,
      y = y
      )
  return(list(
    plot = plot,
    correlation = cor,
    cor_test = cor_test,
    linear_regression = summary_lm,
    generalized_additive_model = summary_gam
    )
  )
}

## Apply the Function

# Calories Estimation Features
bivariate_numerical(data = data_train,x = "calories_met_estimation",y = "Calories")    # lm R-squared:  0.9126 Gam Deviance explained =   94%
bivariate_numerical(data = data_train,x = "calories_met_ffmi_adjusted",y = "Calories") # lm R-squared:  0.9265 Gam Deviance explained = 95.3%
bivariate_numerical(data = data_train,x = "cal_thermodynamics",y = "Calories")         # lm R-squared:  0.468  Gam Deviance explained = 49.2%
bivariate_numerical(data = data_train,x = "keytel_theoretical",y = "Calories")         # lm R-squared:  0.3809 Gam Deviance explained = 38.9%
bivariate_numerical(data = data_train,x = "metabolic_vo2_cal",y = "Calories")          # lm R-squared:  0.9214 Gam Deviance explained = 92.7%
bivariate_numerical(data = data_train,x = "total_burn_from_body_heat",y = "Calories")  # lm R-squared:  0.806  Gam Deviance explained = 81.6%

# Derived Features
bivariate_numerical(data = data_train,x = "met",y = "Calories")               # lm R-squared:  0.8469 # Gam Deviance explained =   88%
bivariate_numerical(data = data_train,x = "vo2",y = "Calories")               # lm R-squared:  0.8469 # Gam Deviance explained =   88%
bivariate_numerical(data = data_train,x = "met_ffmi_adjusted",y = "Calories") # lm R-squared:  0.8245 # Gam Deviance explained = 85.3%
bivariate_numerical(data = data_train,x = "effort",y = "Calories")            # lm R-squared:  0.8471 # Gam Deviance explained = 87.5%
bivariate_numerical(data = data_train,x = "delta_t",y = "Calories")           # lm R-squared: 0.6908  # Gam Deviance explained = 82.4%

# Original Features
bivariate_numerical(data = data_train,x = "Height",y = "Calories")     # lm p-value: 0.8243 XXX
bivariate_numerical(data = data_train,x = "Heart_Rate",y = "Calories") # lm R-squared:  0.828   # Gam Deviance explained = 86.1%
bivariate_numerical(data = data_train,x = "Weight",y = "Calories")     # lm p-value: 0.2787 XXX
bivariate_numerical(data = data_train,x = "Body_Temp",y = "Calories")  # lm R-squared:  0.6908  # Gam Deviance explained = 82.4%
bivariate_numerical(data = data_train,x = "Age",y = "Calories")        # lm R-squared:  0.02156 # Gam Deviance explained = 2.53%
bivariate_numerical(data = data_train,x = "Duration",y = "Calories")   # lm R-squared:  0.9201  # Gam Deviance explained = 93.9%

# Bivariate target ~ Categorical
bivariate_categorical <- function(data = data_train, x = "Calories", y) {
  
  # Summarize the data 
  summary_data <- data %>%
    group_by(.data[[y]]) %>%
    summarise(mean_value = mean(.data[[x]], na.rm = TRUE), .groups = "drop")
  
  # Barplot
  plot <- ggplot(summary_data, aes(x = mean_value, y = .data[[y]], fill = .data[[y]])) +
    geom_col() +
    geom_text(aes(label = round(mean_value, 2)), hjust = 1.1, color = "white",size = 10) +
    theme_minimal() +
    labs(
      title = paste("Mean", x, "by", y),
      x = paste("Mean", x),
      y = ""
    ) +
    theme(
      legend.position = "none")
  
  # Parametric and Non-parametric test
  t_test <- t.test(data[[x]] ~ data[[y]])
  wilcox_test <- wilcox.test(data[[x]] ~ data[[y]])
  
  return(list(
    plot = plot,
    t_test = t_test,
    wilcox_test = wilcox_test
  ))
}

# Apply the function     
bivariate_categorical(y = "classification_BMI_Prime") # Wilcoxon rank sum test p-value = 0.02651
bivariate_categorical(y = "classification_PI")        # Wilcoxon rank sum test p-value = 0.1807 XXX
bivariate_categorical(y = "Sex")                      # Wilcoxon rank sum test p-value = 0.906 XXX

table(data_train$classification_BMI,data_train$Sex)

# Chi-squared test classification_BMI and Sex
chisq.test(x = data_train$Sex,y = data_train$classification_BMI)

## Correlation plot 
data_train %>%
  select(where(is.numeric))%>%
  cor()%>%
  ggcorrplot(lab = FALSE) 

#### Multivariate Analysis ####

## PCA 

# Perform PCA
pca_result <- prcomp(data_train %>% select(where(is.numeric)) %>% select(-Calories),
                     scale. = TRUE,center = TRUE)

# Scree plot 
screeplot(pca_result, type = "lines", main = "Scree Plot", col = "blue")

# Extract only the first 5 principal components
pca_data <- as.data.frame(pca_result$x[, 1:5])  

# Combine the PCA components with the target variable (Calories)
pca_data_with_target <- cbind(pca_data, Calories = data_train$Calories,Sex = data_train$Sex)

# Linear regression with PC1 - PC5
lm_pca_5 <- lm(formula = Calories ~ PC1 + PC2 + PC3 + PC4 + PC5 + Sex,data = pca_data_with_target)
summary(lm_pca_5)

# Look at the predicted and Actual  Values
cor(lm_pca_5$fitted.values,data_train$Calories)
plot(lm_pca_5$fitted.values,data_train$Calories)
rmsle(lm_pca_5$fitted.values,data_train$Calories)

# GAM with PCA 
gam_pca_5 <- gam(formula = Calories ~ s(PC1) + s(PC2) + s(PC3) + s(PC4) + s(PC5)+ Sex,data = pca_data_with_target)

# Look at the predicted and Actual Values
cor(gam_pca_5$fitted.values,data_train$Calories)
plot(gam_pca_5$fitted.values,data_train$Calories)

#### Model Based Analysis ####

# Make a Random Forest Model 
random_forest <- ranger(
  Calories ~ .,data = data_train,
  importance = "permutation")

# Plot importance
barplot(sort(random_forest$variable.importance),
        las = 1,
        col = "skyblue",     
        main = "Permutation Importance",
        horiz = TRUE,
        cex.names = 0.7)

# Check the results
cor(data_train$Calories,random_forest$predictions) #0.9979955
plot(data_train$Calories,random_forest$predictions)
rmsle(random_forest$predictions,predicted = data_train$Calories) # Initial start 0.0650426

# Level 1 learner lm
ll <- lm(Calories ~ random_forest$predictions,data = data_train)
summary(ll)
rmsle(actual = data_train$Calories,predicted = ll$fitted.values) # 0.06483927

# Level 1 learner gam
gam <-gam(Calories ~ s(random_forest$predictions),data = data_train)
summary(gam)
rmsle(actual = data_train$Calories,predicted = gam$fitted.values) # 0.06469167

