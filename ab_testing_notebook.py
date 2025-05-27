# A/B Testing Analysis: E-commerce Checkout Optimization
# Jupyter Notebook Version with Detailed Explanations

"""
This notebook demonstrates a comprehensive A/B testing analysis for optimizing 
an e-commerce checkout process. It showcases statistical rigor, business acumen,
and practical implementation skills valued by data science employers.

Table of Contents:
1. Introduction and Problem Statement
2. Experimental Design and Power Analysis
3. Data Generation and Quality Checks
4. Primary Metric Analysis
5. Secondary Metrics and Segmentation
6. Advanced Statistical Methods
7. Business Impact and ROI
8. Visualizations and Dashboard
9. Recommendations and Next Steps
"""

# ============================================================================
# CELL 1: Setup and Introduction
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import beta, norm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats.power import ttest_power
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Set styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

print("🚀 A/B Testing Analysis: E-commerce Checkout Optimization")
print("=" * 70)

# ============================================================================
# CELL 2: Problem Statement and Experimental Design
# ============================================================================

"""
# 1. Problem Statement 🎯

## Business Challenge
Our e-commerce platform has a **68% checkout abandonment rate**, costing an estimated 
**$2.3M annually** in lost revenue. The current 4-step checkout process is too complex 
and causes friction for users, especially on mobile devices.

## Hypothesis
**H₀**: The new streamlined checkout design has no effect on conversion rate
**H₁**: The new streamlined checkout design increases conversion rate by at least 2%

## Test Design
- **Control (A)**: Original 4-step checkout process
- **Treatment (B)**: Streamlined 2-step checkout with progress indicators
- **Primary Metric**: Conversion rate (purchases/sessions)
- **Secondary Metrics**: Revenue per user, checkout time, user satisfaction
- **Duration**: 21 days
- **Traffic Split**: 50/50 randomized assignment

## Success Criteria
- Minimum 2% absolute increase in conversion rate
- Statistical significance (p < 0.05)
- No decrease in average order value
- Business impact > $1M annually
"""

# Key experimental parameters
BASELINE_CONVERSION = 0.1177  # Current conversion rate
MIN_DETECTABLE_EFFECT = 0.02  # 2% absolute increase
ALPHA = 0.05  # Significance level
POWER = 0.80  # Statistical power
TEST_DURATION_DAYS = 21

print(f"📊 Experimental Parameters:")
print(f"   Baseline conversion rate: {BASELINE_CONVERSION:.1%}")
print(f"   Minimum detectable effect: {MIN_DETECTABLE_EFFECT:.1%}")
print(f"   Significance level (α): {ALPHA}")
print(f"   Statistical power: {POWER}")
print(f"   Test duration: {TEST_DURATION_DAYS} days")

# ============================================================================
# CELL 3: Power Analysis and Sample Size Calculation
# ============================================================================

"""
# 2. Sample Size Calculation 📏

Before running any experiment, we need to determine the required sample size to detect
our minimum effect with adequate statistical power.

The sample size calculation ensures we have enough data to:
- Detect a meaningful business impact (2% conversion lift)
- Achieve 80% statistical power
- Maintain 5% significance level (Type I error rate)
"""

def calculate_sample_size(baseline_rate, minimum_effect, alpha=0.05, power=0.8):
    """
    Calculate required sample size for A/B test using two-proportion z-test
    
    Parameters:
    -----------
    baseline_rate : float
        Current conversion rate (control group expected rate)
    minimum_effect : float  
        Minimum detectable effect (absolute difference)
    alpha : float
        Significance level (Type I error rate)
    power : float
        Statistical power (1 - Type II error rate)
    
    Returns:
    --------
    int : Required sample size per group
    """
    # Calculate effect size
    p1 = baseline_rate
    p2 = baseline_rate + minimum_effect
    
    # Pooled standard error under null hypothesis
    p_pooled = (p1 + p2) / 2
    se = np.sqrt(2 * p_pooled * (1 - p_pooled))
    
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed test
    z_beta = stats.norm.ppf(power)
    
    # Sample size calculation
    n = ((z_alpha + z_beta) * se / minimum_effect) ** 2
    return int(np.ceil(n))

# Calculate required sample size
sample_size_per_group = calculate_sample_size(
    BASELINE_CONVERSION, MIN_DETECTABLE_EFFECT, ALPHA, POWER
)

print(f"📈 Sample Size Analysis:")
print(f"   Required sample size per group: {sample_size_per_group:,}")
print(f"   Total required sample size: {sample_size_per_group * 2:,}")
print(f"   Daily traffic needed: {(sample_size_per_group * 2) // TEST_DURATION_DAYS:,}")

# Power curve visualization
effect_sizes = np.linspace(0.005, 0.04, 50)
sample_sizes = [calculate_sample_size(BASELINE_CONVERSION, effect, ALPHA, POWER) 
                for effect in effect_sizes]

plt.figure(figsize=(10, 6))
plt.plot(effect_sizes * 100, sample_sizes, linewidth=2, color='#2E86C1')
plt.axvline(MIN_DETECTABLE_EFFECT * 100, color='red', linestyle='--', 
            label=f'Target Effect: {MIN_DETECTABLE_EFFECT:.1%}')
plt.axhline(sample_size_per_group, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Minimum Detectable Effect (%)')
plt.ylabel('Required Sample Size per Group')
plt.title('Sample Size vs. Effect Size')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================================
# CELL 4: Data Generation
# ============================================================================

"""
# 3. Data Generation 🔧

For this portfolio project, we'll generate realistic synthetic data that mimics 
real-world A/B testing scenarios. The data includes:

- **User segmentation**: Device type, user type, traffic source
- **Realistic conversion patterns**: Mobile users convert less, returning users convert more
- **Treatment effects**: Bigger improvements for mobile and new users
- **Business metrics**: Order values, checkout times, revenue

This approach demonstrates how to work with realistic, messy data while maintaining
statistical validity.
"""

def generate_ab_test_data(n_control=15680, n_treatment=15680):
    """
    Generate realistic A/B test data with user segments and treatment effects
    
    Parameters:
    -----------
    n_control : int
        Number of users in control group
    n_treatment : int  
        Number of users in treatment group
        
    Returns:
    --------
    pd.DataFrame : Generated experiment data
    """
    
    print(f"🏭 Generating synthetic experiment data...")
    print(f"   Control group size: {n_control:,}")
    print(f"   Treatment group size: {n_treatment:,}")
    
    # Base conversion rates and metrics
    control_conversion_rate = 0.1177
    treatment_conversion_rate = 0.1375
    base_order_value = 85.40
    control_checkout_time = 284  # seconds
    treatment_checkout_time = 198  # seconds
    
    data = []
    
    # Generate control group data
    for i in range(n_control):
        user_id = f"control_{i}"
        
        # User characteristics
        device_type = np.random.choice(['mobile', 'desktop', 'tablet'], 
                                     p=[0.6, 0.35, 0.05])
        user_type = np.random.choice(['new_user', 'returning_user'], p=[0.4, 0.6])
        traffic_source = np.random.choice(['organic', 'paid', 'direct', 'social'], 
                                        p=[0.4, 0.3, 0.2, 0.1])
        
        # Segment-based conversion adjustment
        if device_type == 'mobile':
            conv_multiplier = 0.78 if user_type == 'new_user' else 0.85
        elif device_type == 'desktop':
            conv_multiplier = 1.28 if user_type == 'new_user' else 1.35
        else:  # tablet
            conv_multiplier = 0.99
            
        # Calculate conversion
        adjusted_rate = control_conversion_rate * conv_multiplier
        converted = np.random.binomial(1, min(adjusted_rate, 1.0))
        
        # Order value (if converted)
        order_value = 0
        if converted:
            # Gamma distribution for realistic order value distribution
            order_value = np.random.gamma(2, base_order_value/2)
            
        # Checkout time (all users attempt checkout)
        checkout_time = np.random.gamma(4, control_checkout_time/4)
        
        data.append({
            'user_id': user_id,
            'variant': 'control',
            'device_type': device_type,
            'user_type': user_type,
            'traffic_source': traffic_source,
            'converted': converted,
            'order_value': order_value,
            'checkout_time': checkout_time
        })
    
    # Generate treatment group data
    for i in range(n_treatment):
        user_id = f"treatment_{i}"
        
        device_type = np.random.choice(['mobile', 'desktop', 'tablet'], 
                                     p=[0.6, 0.35, 0.05])
        user_type = np.random.choice(['new_user', 'returning_user'], p=[0.4, 0.6])
        traffic_source = np.random.choice(['organic', 'paid', 'direct', 'social'], 
                                        p=[0.4, 0.3, 0.2, 0.1])
        
        # Treatment shows bigger improvement for mobile and new users
        if device_type == 'mobile':
            conv_multiplier = 1.02 if user_type == 'new_user' else 1.12
        elif device_type == 'desktop':
            conv_multiplier = 1.38 if user_type == 'new_user' else 1.46
        else:  # tablet
            conv_multiplier = 1.03
            
        adjusted_rate = treatment_conversion_rate * conv_multiplier
        converted = np.random.binomial(1, min(adjusted_rate, 1.0))
        
        order_value = 0
        if converted:
            # Slightly higher order value for treatment (better UX)
            order_value = np.random.gamma(2, (base_order_value * 1.02)/2)
            
        # Faster checkout time for treatment
        checkout_time = np.random.gamma(3, treatment_checkout_time/3)
        
        data.append({
            'user_id': user_id,
            'variant': 'treatment',
            'device_type': device_type,
            'user_type': user_type,
            'traffic_source': traffic_source,
            'converted': converted,
            'order_value': order_value,
            'checkout_time': checkout_time
        })
    
    return pd.DataFrame(data)

# Generate the experiment dataset
df = generate_ab_test_data()

print(f"✅ Dataset generated successfully!")
print(f"   Total users: {len(df):,}")
print(f"   Columns: {list(df.columns)}")
print(f"   Date range: {TEST_DURATION_DAYS} days")

# Display first few rows
print("\n📋 Sample Data:")
print(df.head(10))

# Basic statistics
print("\n📊 Dataset Summary:")
print(df.describe())

# ============================================================================
# CELL 5: Data Quality Checks
# ============================================================================

"""
# 4. Data Quality Assurance 🔍

Before analyzing results, we must validate our experiment data for common issues:

## Sample Ratio Mismatch (SRM)
Ensures the traffic split is as expected (50/50). Deviations can indicate:
- Technical issues with randomization
- Selection bias
- Data collection problems

## Data Completeness
Checks for missing values, outliers, and data consistency.

## Randomization Check
Verifies that user characteristics are balanced between groups.
"""

def check_sample_ratio_mismatch(df, expected_ratio=0.5, alpha=0.001):
    """
    Test for Sample Ratio Mismatch using Chi-square test
    
    Parameters:
    -----------
    df : pd.DataFrame
        Experiment data
    expected_ratio : float
        Expected proportion for each group (0.5 for 50/50 split)
    alpha : float
        Significance level for SRM test (typically 0.001)
    """
    variant_counts = df['variant'].value_counts()
    control_count = variant_counts.get('control', 0)
    treatment_count = variant_counts.get('treatment', 0)
    total = control_count + treatment_count
    
    expected_control = total * expected_ratio
    expected_treatment = total * (1 - expected_ratio)
    
    # Chi-square test
    chi2_stat = ((control_count - expected_control)**2 / expected_control + 
                 (treatment_count - expected_treatment)**2 / expected_treatment)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    
    print(f"🔍 Sample Ratio Mismatch Check:")
    print(f"   Control: {control_count:,} ({control_count/total:.1%})")
    print(f"   Treatment: {treatment_count:,} ({treatment_count/total:.1%})")
    print(f"   Expected split: {expected_ratio:.1%} / {1-expected_ratio:.1%}")
    print(f"   Chi-square statistic: {chi2_stat:.4f}")
    print(f"   P-value: {p_value:.6f}")
    print(f"   Result: {'✅ PASS' if p_value > alpha else '❌ FAIL - INVESTIGATE'}")
    
    return p_value > alpha

# Perform SRM check
srm_passed = check_sample_ratio_mismatch(df)

# Check for missing data
print(f"\n📊 Data Completeness Check:")
missing_data = df.isnull().sum()
print(f"   Missing values per column:")
for col, missing in missing_data.items():
    print(f"   {col}: {missing} ({missing/len(df):.1%})")

# Check for outliers in order values
converted_users = df[df['converted'] == 1]
if len(converted_users) > 0:
    q99 = converted_users['order_value'].quantile(0.99)
    outliers = (converted_users['order_value'] > q99).sum()
    print(f"\n💰 Order Value Analysis:")
    print(f"   99th percentile: ${q99:.2f}")
    print(f"   Potential outliers (>99th percentile): {outliers}")

# Randomization balance check
print(f"\n⚖️ Randomization Balance Check:")
for column in ['device_type', 'user_type', 'traffic_source']:
    balance_check = pd.crosstab(df[column], df['variant'], normalize='columns')
    print(f"\n   {column.title()} Distribution:")
    print(balance_check.round(3))

# ============================================================================
# CELL 6: Primary Metric Analysis
# ============================================================================

"""
# 5. Primary Metric Analysis 📈

The primary metric is **conversion rate** - the percentage of users who complete a purchase.
This is our key business metric and the main outcome we're trying to improve.

## Statistical Tests Applied:
1. **Two-proportion z-test**: Tests if conversion rates differ significantly
2. **Confidence intervals**: Provides range of plausible values for the true effect
3. **Effect size calculations**: Measures practical significance
"""

def analyze_conversion_rate(df):
    """
    Comprehensive conversion rate analysis with statistical tests
    """
    
    # Calculate conversion metrics by variant
    conversion_summary = df.groupby('variant').agg({
        'converted': ['count', 'sum', 'mean'],
        'order_value': 'sum'
    }).round(4)
    
    # Flatten column names
    conversion_summary.columns = ['sessions', 'conversions', 'conversion_rate', 'total_revenue']
    conversion_summary['revenue_per_user'] = (
        conversion_summary['total_revenue'] / conversion_summary['sessions']
    )
    
    print("🎯 PRIMARY METRIC: CONVERSION RATE")
    print("=" * 50)
    print(conversion_summary)
    print()
    
    # Extract values for statistical testing
    control_conv = conversion_summary.loc['control', 'conversions']
    control_sessions = conversion_summary.loc['control', 'sessions']
    treatment_conv = conversion_summary.loc['treatment', 'conversions']
    treatment_sessions = conversion_summary.loc['treatment', 'sessions']
    
    control_rate = control_conv / control_sessions
    treatment_rate = treatment_conv / treatment_sessions
    
    # Two-proportion z-test
    z_stat, p_value = proportions_ztest(
        [treatment_conv, control_conv], 
        [treatment_sessions, control_sessions]
    )
    
    # Confidence intervals (95%)
    control_ci = proportion_confint(control_conv, control_sessions, alpha=0.05)
    treatment_ci = proportion_confint(treatment_conv, treatment_sessions, alpha=0.05)
    
    # Effect size calculations
    absolute_lift = treatment_rate - control_rate
    relative_lift = (treatment_rate / control_rate - 1) * 100
    
    # Statistical power (post-hoc)
    observed_effect = absolute_lift
    pooled_p = (control_conv + treatment_conv) / (control_sessions + treatment_sessions)
    pooled_se = np.sqrt(2 * pooled_p * (1 - pooled_p) / min(control_sessions, treatment_sessions))
    observed_power = 1 - stats.norm.cdf(1.96 - observed_effect / pooled_se)
    
    print(f"📊 Statistical Test Results:")
    print(f"   Control conversion rate:    {control_rate:.3%}")
    print(f"   Treatment conversion rate:  {treatment_rate:.3%}")
    print(f"   95% CI Control:             [{control_ci[0]:.3%}, {control_ci[1]:.3%}]")
    print(f"   95% CI Treatment:           [{treatment_ci[0]:.3%}, {treatment_ci[1]:.3%}]")
    print()
    print(f"🎯 Effect Size:")
    print(f"   Absolute lift:              +{absolute_lift:.3%}")
    print(f"   Relative lift:              +{relative_lift:.1f}%")
    print()
    print(f"🧪 Statistical Significance:")
    print(f"   Z-statistic:                {z_stat:.4f}")
    print(f"   P-value:                    {p_value:.6f}")
    print(f"   Significant (α=0.05):       {'✅ YES' if p_value < 0.05 else '❌ NO'}")
    print(f"   Observed power:             {observed_power:.1%}")
    
    return conversion_summary, p_value, absolute_lift, relative_lift

# Run conversion rate analysis
conversion_results, conv_p_value, abs_lift, rel_lift = analyze_conversion_rate(df)

# Visualization of conversion rates
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Conversion rates with confidence intervals
variants = ['Control', 'Treatment']
rates = [conversion_results.loc['control', 'conversion_rate'], 
         conversion_results.loc['treatment', 'conversion_rate']]
colors = ['#FF6B6B', '#4ECDC4']

bars = ax1.bar(variants, rates, color=colors, alpha=0.8, capsize=5)
ax1.set_title('Conversion Rate by Variant', fontsize=14, fontweight='bold')
ax1.set_ylabel('Conversion Rate')
ax1.set_ylim(0, max(rates) * 1.2)

# Add value labels
for bar, rate in zip(bars, rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
            f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')

# Revenue per user
rpu_values = [conversion_results.loc['control', 'revenue_per_user'], 
              conversion_results.loc['treatment', 'revenue_per_user']]
bars2 = ax2.bar(variants, rpu_values, color=colors, alpha=0.8)
ax2.set_title('Revenue per User', fontsize=14, fontweight='bold')
ax2.set_ylabel('Revenue per User ($)')

for bar, rpu in zip(bars2, rpu_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'${rpu:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# CELL 7: Segmentation Analysis
# ============================================================================

"""
# 6. Segmentation Analysis 📱💻

Understanding how different user segments respond to the treatment helps us:
- Identify which users benefit most from the change
- Optimize future iterations for specific segments  
- Make targeted implementation decisions
- Understand the underlying mechanisms of the effect

We'll analyze performance across:
- **Device type**: Mobile, desktop, tablet users
- **User type**: New vs. returning users
- **Traffic source**: How users arrived at the site
"""

def segmentation_analysis(df, segment_col, min_sample_size=100):
    """
    Analyze treatment effect across different user segments
    
    Parameters:
    -----------
    df : pd.DataFrame
        Experiment data
    segment_col : str
        Column name for segmentation
    min_sample_size : int
        Minimum sample size per segment for statistical testing
    """
    
    print(f"📱 SEGMENTATION ANALYSIS: {segment_col.upper()}")
    print("=" * 60)
    
    segments = df[segment_col].unique()
    segment_results = []
    
    for segment in segments:
        segment_data = df[df[segment_col] == segment]
        
        # Split by variant
        control_segment = segment_data[segment_data['variant'] == 'control']
        treatment_segment = segment_data[segment_data['variant'] == 'treatment']
        
        # Calculate metrics
        control_conversions = control_segment['converted'].sum()
        control_sessions = len(control_segment)
        treatment_conversions = treatment_segment['converted'].sum()
        treatment_sessions = len(treatment_segment)
        
        if (control_sessions >= min_sample_size and treatment_sessions >= min_sample_size and
            control_conversions >= 5 and treatment_conversions >= 5):
            
            control_rate = control_conversions / control_sessions
            treatment_rate = treatment_conversions / treatment_sessions
            
            # Statistical test
            z_stat, p_value = proportions_ztest(
                [treatment_conversions, control_conversions], 
                [treatment_sessions, control_sessions]
            )
            
            absolute_lift = treatment_rate - control_rate
            relative_lift = (treatment_rate / control_rate - 1) * 100 if control_rate > 0 else 0
            
            segment_results.append({
                'segment': segment,
                'control_sessions': control_sessions,
                'treatment_sessions': treatment_sessions,
                'control_rate': control_rate,
                'treatment_rate': treatment_rate,
                'absolute_lift': absolute_lift,
                'relative_lift': relative_lift,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
            
            significance = '✅ SIG' if p_value < 0.05 else '❌ NS'
            print(f"{segment:12} | Control: {control_rate:.1%} | Treatment: {treatment_rate:.1%} | "
                  f"Lift: {relative_lift:+.1f}% | p={p_value:.3f} | {significance}")
        else:
            print(f"{segment:12} | Insufficient sample size for reliable testing")
    
    print()
    return pd.DataFrame(segment_results)

# Analyze different segments
device_segments = segmentation_analysis(df, 'device_type')
user_segments = segmentation_analysis(df, 'user_type')
traffic_segments = segmentation_analysis(df, 'traffic_source')

# Visualization of segmentation results
if not device_segments.empty:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Device type analysis
    device_plot_data = df.groupby(['device_type', 'variant'])['converted'].mean().unstack()
    device_plot_data.plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    axes[0].set_title('Conversion Rate by Device Type', fontweight='bold')
    axes[0].set_ylabel('Conversion Rate')
    axes[0].legend(['Control', 'Treatment'])
    axes[0].tick_params(axis='x', rotation=45)
    
    # User type analysis
    user_plot_data = df.groupby(['user_type', 'variant'])['converted'].mean().unstack()
    user_plot_data.plot(kind='bar', ax=axes[1], color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    axes[1].set_title('Conversion Rate by User Type', fontweight='bold')
    axes[1].set_ylabel('Conversion Rate')
    axes[1].legend(['Control', 'Treatment'])
    axes[1].tick_params(axis='x', rotation=45)
    
    # Traffic source analysis
    traffic_plot_data = df.groupby(['traffic_source', 'variant'])['converted'].mean().unstack()
    traffic_plot_data.plot(kind='bar', ax=axes[2], color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    axes[2].set_title('Conversion Rate by Traffic Source', fontweight='bold')
    axes[2].set_ylabel('Conversion Rate')
    axes[2].legend(['Control', 'Treatment'])
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# CELL 8: Bayesian Analysis
# ============================================================================

"""
# 7. Bayesian Analysis 🔮

While frequentist methods tell us if there's a statistically significant difference,
Bayesian analysis provides intuitive probabilistic answers to business questions:

- **"What's the probability that Treatment is better than Control?"**
- **"What's the expected lift and its uncertainty?"**
- **"What's the risk of a negative impact?"**

This approach is increasingly popular in industry because it directly answers
business questions and provides better decision-making frameworks.
"""

def bayesian_ab_test(control_conversions, control_sessions, treatment_conversions, treatment_sessions):
    """
    Bayesian A/B test analysis using Beta-Binomial model
    
    Returns probability that treatment is better and credible intervals
    """
    
    print("🔮 BAYESIAN ANALYSIS")
    print("=" * 50)
    
    # Prior parameters (non-informative uniform prior)
    alpha_prior = 1
    beta_prior = 1
    
    # Posterior parameters (Beta distribution)
    alpha_control = alpha_prior + control_conversions
    beta_control = beta_prior + control_sessions - control_conversions
    
    alpha_treatment = alpha_prior + treatment_conversions
    beta_treatment = beta_prior + treatment_sessions - treatment_conversions
    
    # Sample from posterior distributions
    n_samples = 100000
    control_samples = np.random.beta(alpha_control, beta_control, n_samples)
    treatment_samples = np.random.beta(alpha_treatment, beta_treatment, n_samples)
    
    # Calculate lift distributions
    lift_samples = treatment_samples - control_samples
    relative_lift_samples = (treatment_samples / control_samples - 1) * 100
    
    # Key Bayesian metrics
    prob_treatment_better = np.mean(treatment_samples > control_samples)
    prob_significant_lift = np.mean(lift_samples > 0.01)  # >1% absolute lift
    prob_negative_impact = np.mean(lift_samples < 0)
    
    # Credible intervals (Bayesian equivalent of confidence intervals)
    lift_ci = np.percentile(lift_samples, [2.5, 97.5])
    relative_lift_ci = np.percentile(relative_lift_samples, [2.5, 97.5])
    
    print(f"🎯 Bayesian Results:")
    print(f"   Probability Treatment > Control:     {prob_treatment_better:.1%}")
    print(f"   Probability of >1% absolute lift:    {prob_significant_lift:.1%}")
    print(f"   Risk of negative impact:             {prob_negative_impact:.1%}")
    print()
    print(f"📊 Expected Effects:")
    print(f"   Expected absolute lift:              {np.mean(lift_samples):.3%}")
    print(f"   95% Credible Interval:               [{lift_ci[0]:.3%}, {lift_ci[1]:.3%}]")
    print(f"   Expected relative lift:              {np.mean(relative_lift_samples):.1f}%")
    print(f"   95% Credible Interval:               [{relative_lift_ci[0]:.1f}%, {relative_lift_ci[1]:.1f}%]")
    
    return {
        'prob_treatment_better': prob_treatment_better,
        'prob_significant_lift': prob_significant_lift,
        'prob_negative_impact': prob_negative_impact,
        'expected_lift': np.mean(lift_samples),
        'lift_ci': lift_ci,
        'lift_samples': lift_samples,
        'relative_lift_samples': relative_