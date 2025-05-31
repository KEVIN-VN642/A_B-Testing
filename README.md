# A/B Testing Analysis for E-commerce Checkout Optimization

## 1. Project Introduction
### 1.1 Overview A/B Testing
A/B testing, also known as split testing, is a powerful method that allows organizations to compare two or more variations of a webpage, app feature, or campaign to determine which performs better. By systematically testing changes with real users, businesses can move beyond assumptions and intuition, by relying on empirical evidence to guide improvements. This not only reduces risk but also optimizes outcomes—boosting conversion rates, user engagement, and overall return on investment. 

This project analyzes a comprehensive A/B testing experiment conducted on an e-commerce platform to optimize the checkout process. The study examines the impact of design changes on conversion rates, revenue per user, and other metrics.

### 1.2 Business Problem

The current checkout process has a 38% abandonment rate, costing the company an estimated $1.3M annually lost in revenue. The management want to make changes in this process to boost conversion rate, average order value, and user experience.


## 2. Experimental Design
The current checkout process has four steps, team members want to streamline it to two steps only. Here is the setup for the experiment
### Test Variants:
- **Control (A)**: Original 4-step checkout process
- **Treatment (B)**: Use 2-step checkout process
### Hypothesis:
- **H₀**: The new checkout design has no effect on conversion rate  
- **H₁**: The new checkout design increases conversion rate by at least 2%
### Experiment parameters:
- baseline_conversion = 0.1 -  Current conversion rate 
- minimum_detectable_effect = 0.02  - 2% absolute increase 
- alpha = 0.05  - Significance level 
- power = 0.80  - Statistical power

### **Success Metrics**:
- Primary: Conversion rate (purchases/sessions)
- Secondary: Revenue per user, time to checkout completion

### Sample Size Calculation

- **Sample Size**: Required sample size is 3843 users per group.
- **Test Duration**: 16 days  (base on current traffic at 470 users/day)
- **Traffic Split**: 50/50 randomized assignment

### Randomization & Controls
- **Stratified randomization** by device type (mobile/desktop) and user type (new/returning)
- **Consistent user experience** - users remain in assigned group throughout test period

## 3. Data Quality Check
Before analyzing results, we must validate the experiment data for common issues which might mislead the result.
- **Sample Ratio Mismatch (SRM) Check**: Data split betweeen control and treatment should not be much different from expected split.
- **Sample size**: Make sure number of samples in each group exceeds the required sample size
- **Data Completeness**: Checks for missing values, outliers, and data consistency.
- **Randomization Check**: Verifies that user characteristics are balanced between groups.

### 3.1 Sample Size Check
This makes sure number of samples in each group exceeds the required number. There are 4957 users in control group and 5043 users in treatment group which is significant larger than required number, so the sample size requirement meets.

### 3.2  Sample Ratio Mismatch Check
Ensures the traffic split is as expected (50/50) before doing analysis. SRM protects us from making business decisions based on flawed data. Deviations can come froms various sources:
- Technical issues: randomization algorithm bugs, client-side tracking failures
- Selection bias: geographic or temporal biases in assignment
- Data collection problems: missing data from one variant, logging errors <br>
The result show that the experiment data align well with design split ratio (50/50) <br>
  * Control Count: 4,957 (49.6%) <br>
  * Treatment Count: 5,043 (50.4%)<br>
  * Expected split: 50.0% / 50.0%<br>
  * Chi-square statistic: 0.7396<br>
  * P-value: 0.389789 <br>
  * Result: ✅ PASS

### 3.3 Randomization Check
We expect that user characteristics are balanced between groups. The bar plot shows the data is quite balanced in each category.
![Randomization](https://github.com/KEVIN-VN642/A_B-Testing/blob/main/images/Randomization_Check.png)
![Randomization](https://github.com/KEVIN-VN642/A_B-Testing/blob/main/images/Randomization_Check1.png)

### 3.4 Boxplot and Data Distribution
This visualize the order value to explore data distribution and potential outliers.
![boxplot](https://github.com/KEVIN-VN642/A_B-Testing/blob/main/images/boxplot.png)
![distribution](https://github.com/KEVIN-VN642/A_B-Testing/blob/main/images/distribution.png)

## 4. Statistical Analysis

In this section we will perform statistical analyis on the primary metric **conversion rate** - the percentage of users who complete a purchase. Conversion rate is our key metric and the main outcome we're trying to improve. We also analyze other secondary metrics such as effect size, revenue per user. 

### The following statistical tests are applied:
- **Two-proportion z-test**: Tests if conversion rates differ significantly
- **Confidence intervals**: Provides range of plausible values for the true effect
- **Effect size calculations**: Measures practical significance<br>
### Results: <br>

**Statistical Test Results:** <br>
   Control conversion rate:    10.309% <br>
   Treatment conversion rate:  14.614% <br>
   95% CI Control:             [9.462%, 11.155%] <br>
   95% CI Treatment:           [13.639%, 15.589%] <br>

**Effect Size:** <br>
   Absolute lift:              +4.306% <br>
   Relative lift:              +41.8% <br>
   Absolute lift CI:           3.01% to 5.60%
   
**Statistical Significance:** <br>
   Z-statistic:                6.5138 <br>
   P-value:                    0.000000 <br>
   Significant (α=0.05):       ✅ YES <br>
   Observed power:             100.0% <br>
**Revenue Per User**: <br>
   Control:                    $21.79 <br>
   Treatment:                  $31.56 <br>
   Revenue lift:               44.79% <br>

![metrics](https://github.com/KEVIN-VN642/A_B-Testing/blob/main/images/key_metrics.png)

## 5. Segmentation Analysis

#### Device Type
| Segment         | Control Rate | Treatment Rate |   Lift   |   p   | Significant |
|-----------------|--------------|----------------|----------|-------|-------------|
|      Mobil      |    8.8%      |    13.2%       |  46.6%   | 0.000 |   ✅ SIG    
|     Desktop     |    13.2%     |    18.0%       |  36.3%   | 0.000 |   ✅ SIG
|      Tablet     |    8.1%      |    11.3%       |  40.0%   | 0.224 |   ❌ NS


#### User Type
|    Segment      | Control Rate | Treatment Rate |   Lift   |   p   | Significant |
|-----------------|--------------|----------------|----------|-------|-------------|
| returning_user  |    11.0%     |    15.6%       |  41.9%   | 0.000 |   ✅ SIG    
| new_User        |    9.3%      |    13.2%       |  42.4%   | 0.000 |   ✅ SIG

#### Traffic Source
|    Segment      | Control Rate | Treatment Rate |   Lift   |   p   | Significant |
|-----------------|--------------|----------------|----------|-------|-------------|
|direct       | 10.2% | 14.3% | 40.3% | 0.006 | ✅ SIG
|organic      | 10.7% | 14.0% | 31.0% | 0.001 | ✅ SIG
|paid         | 10.0% | 15.6% | 55.1% | 0.000 | ✅ SIG
|social       | 9.8% | 14.7% | 49.5% | 0.017 | ✅ SIG

**Key Insights**:
- **Mobile users** show the largest improvement (+28.3% lift)
- **New users** benefit more than returning users
Treatment is particularly effective for mobile-first customer segments

![Segmentation](https://github.com/KEVIN-VN642/A_B-Testing/blob/main/images/segmentation.png)

## 6. Recommendation
Decision Criteria Scorecard:<br>
   Statistical Significance  ✅ PASS     (p = 0.0000)<br>
   Practical Significance    ✅ PASS     (4.31% lift, confidence interval [3.01, 5.60])<br>
   Business Impact           ✅ PASS     ($888,275)<br>
TOTAL SCORE: 3/3 criteria met
Since all criteria pass the requirement, the practical significance is outside and smaller of confidence interval for absolute lift. We decide to launch the change with high level of confidence.

