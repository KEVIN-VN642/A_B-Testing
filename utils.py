
import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

def calculate_sample_size(baseline_rate, minimum_effect, alpha=0.05, power=0.8):
    """
    Calculate required sample size for A/B test using two-proportion z-test
    
    Parameters:
    -----------
    baseline_rate: float, Current conversion rate (control group expected rate)
    minimum_effect: float, Minimum detectable effect (absolute difference)
    alpha: float, Significance level (Type I error rate)
    power: float, Statistical power (1 - Type II error rate)
    
    Returns:
    --------
    int : Required sample size per group
    """
    # Calculate effect size
    p1 = baseline_rate
    p2 = baseline_rate + minimum_effect
    
    # Pooled standard error under null hypothesis
    p_pooled = (p1 + p2) / 2

    # Note that the formula for standard error is:
    # SE = sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    # For equal sample sizes, n1 = n2 = n, so we can simplify to:
    # SE = sqrt(2 * p_pooled * (1 - p_pooled) / n) = se_coefficient / sqrt(n)

    # Standard error coefficient
    se_coefficient = np.sqrt(2 * p_pooled * (1 - p_pooled))
    
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed test
    z_beta = stats.norm.ppf(power)
    
    # Sample size calculation
    n = ((z_alpha + z_beta) * se_coefficient / minimum_effect) ** 2
    return int(np.ceil(n))

def check_sample_ratio_mismatch(df, expected_ratio=0.5, alpha=0.001):
    """
    Test for Sample Ratio Mismatch using Chi-square test
    
    Parameters:
    -----------
    df : pd.DataFrame, Experiment data
    expected_ratio : float, Expected proportion for each group (0.5 for 50/50 split)
    alpha : float, Significance level for SRM test (typically 0.001)
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
    
    print(f"   Sample Ratio Mismatch Check:")
    print(f"   Control: {control_count:,} ({control_count/total:.1%})")
    print(f"   Treatment: {treatment_count:,} ({treatment_count/total:.1%})")
    print(f"   Expected split: {expected_ratio:.1%} / {1-expected_ratio:.1%}")
    print(f"   Chi-square statistic: {chi2_stat:.4f}")
    print(f"   P-value: {p_value:.6f}")
    print(f"   Result: {'✅ PASS' if p_value > alpha else '❌ FAIL - INVESTIGATE'}")

    return p_value > alpha


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
    
    # Effect size calculations (maginitude of difference)
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
    
    print("💰 Revenue Per User:")
    print(f"   Control:                    ${conversion_summary.loc['control', 'revenue_per_user']:.2f}")
    print(f"   Treatment:                  ${conversion_summary.loc['treatment', 'revenue_per_user']:.2f}")
    print(f"   Revenue lift:               {(conversion_summary.loc['treatment', 'revenue_per_user'] / conversion_summary.loc['control', 'revenue_per_user']-1)*100:.2f}%")



    return conversion_summary, p_value, absolute_lift, relative_lift


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
        'relative_lift_samples': relative_lift_samples
    }



def calculate_business_impact(conversion_results):
    """
    Calculate comprehensive business impact metrics
    """
    
    print("💼 BUSINESS IMPACT ANALYSIS")
    print("=" * 50)
    
    # Business assumptions (adjust based on actual business)
    monthly_sessions = 125000
    annual_sessions = monthly_sessions * 12
    avg_order_value = 85.40
    
    # Implementation costs
    development_cost = 45000
    qa_testing_cost = 8000
    deployment_cost = 3000
    total_implementation = development_cost + qa_testing_cost + deployment_cost
    
    # Ongoing costs
    annual_maintenance = 12000
    monitoring_tools = 2400  # annual
    total_ongoing = annual_maintenance + monitoring_tools
    
    # Current state metrics
    baseline_conversion = conversion_results.loc['control', 'conversion_rate']
    treatment_conversion = conversion_results.loc['treatment', 'conversion_rate']
    
    # Revenue calculations
    baseline_annual_revenue = annual_sessions * baseline_conversion * avg_order_value
    treatment_annual_revenue = annual_sessions * treatment_conversion * avg_order_value
    annual_revenue_lift = treatment_annual_revenue - baseline_annual_revenue
    
    # ROI calculations
    net_annual_benefit = annual_revenue_lift - total_ongoing
    roi_year_1 = net_annual_benefit / total_implementation
    payback_months = total_implementation / (annual_revenue_lift / 12)
    
    # Risk scenarios
    conservative_lift = abs_lift * 0.7  # 30% lower than observed
    conservative_revenue_lift = annual_sessions * conservative_lift * avg_order_value
    
    optimistic_lift = abs_lift * 1.3  # 30% higher than observed  
    optimistic_revenue_lift = annual_sessions * optimistic_lift * avg_order_value
    
    print(f"📊 Current State:")
    print(f"   Monthly sessions:              {monthly_sessions:,}")
    print(f"   Annual sessions:               {annual_sessions:,}")
    print(f"   Current conversion rate:       {baseline_conversion:.2%}")
    print(f"   Current annual revenue:        ${baseline_annual_revenue:,.0f}")
    print()
    
    print(f"🚀 Projected Impact:")
    print(f"   New conversion rate:           {treatment_conversion:.2%}")
    print(f"   Absolute lift:                 +{abs_lift:.2%}")
    print(f"   Relative lift:                 +{rel_lift:.1f}%")
    print(f"   Annual revenue lift:           ${annual_revenue_lift:,.0f}")
    print(f"   Monthly revenue lift:          ${annual_revenue_lift/12:,.0f}")
    print()
    
    print(f"💰 Investment Analysis:")
    print(f"   Development cost:              ${development_cost:,}")
    print(f"   QA & Testing:                  ${qa_testing_cost:,}")
    print(f"   Deployment:                    ${deployment_cost:,}")
    print(f"   Total implementation:          ${total_implementation:,}")
    print(f"   Annual ongoing costs:          ${total_ongoing:,}")
    print()
    
    print(f"📈 ROI Metrics:")
    print(f"   Net annual benefit:            ${net_annual_benefit:,.0f}")
    print(f"   First-year ROI:                {roi_year_1:.1f}x")
    print(f"   Payback period:                {payback_months:.1f} months")
    print()
    
    print(f"🎯 Scenario Analysis:")
    print(f"   Conservative (70% of effect):  ${conservative_revenue_lift:,.0f}")
    print(f"   Expected (observed effect):    ${annual_revenue_lift:,.0f}")
    print(f"   Optimistic (130% of effect):   ${optimistic_revenue_lift:,.0f}")
    
    return {
        'annual_revenue_lift': annual_revenue_lift,
        'roi': roi_year_1,
        'payback_months': payback_months,
        'implementation_cost': total_implementation,
        'conservative_lift': conservative_revenue_lift,
        'optimistic_lift': optimistic_revenue_lift
    }
