
import numpy as np
import pandas as pd
def generate_ab_test_data(n_control=5000, n_treatment=5000):
    """
    Generate realistic A/B test data with user segments and treatment effects
    
    Parameters:
    -----------
    n_control : int, Number of users in control group
    n_treatment : int, Number of users in treatment group
        
    Returns:
    --------
    pd.DataFrame : Generated experiment data
    """
    np.random.seed(42)  # For reproducibility
    print(f" Generating synthetic experiment data...")
    print(f"   Control group size: {n_control:,}")
    print(f"   Treatment group size: {n_treatment:,}")
    
    # Base conversion rates and metrics
    control_conversion_rate = 0.1
    treatment_conversion_rate = 0.12
    base_order_value = 212.40

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
            
        
        data.append({
            'user_id': user_id,
            'variant': 'control',
            'device_type': device_type,
            'user_type': user_type,
            'traffic_source': traffic_source,
            'converted': converted,
            'order_value': order_value
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
            
        
        data.append({
            'user_id': user_id,
            'variant': 'treatment',
            'device_type': device_type,
            'user_type': user_type,
            'traffic_source': traffic_source,
            'converted': converted,
            'order_value': order_value
        })
    df = pd.DataFrame(data)
    df.to_csv('data/ab_test_data.csv', index=False)   

generate_ab_test_data(4957, 5043)  # Adjusted sizes to match original example
