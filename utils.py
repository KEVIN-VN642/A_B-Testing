
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
    
    # calculate confidence interval for absolute lift
    absolute_lift_se = np.sqrt(
        (treatment_rate * (1 - treatment_rate) / treatment_sessions) +
        (control_rate * (1 - control_rate) / control_sessions)
    )
    absolute_lift_ci = (
        absolute_lift - 1.96 * absolute_lift_se,
        absolute_lift + 1.96 * absolute_lift_se
    )
    # Note: The absolute lift CI calculation is a simplified approach.
    # For more accurate CI, you can use bootstrapping or other methods.
    
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
    print(f"   Absolute lift CI:           {absolute_lift_ci[0]:.3%} to {absolute_lift_ci[1]:.3%}")
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

