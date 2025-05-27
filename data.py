# Generate synthetic but realistic A/B test data
import numpy as np
np.random.seed(42)
control_sessions = np.random.binomial(1, 0.1177, 15680)
treatment_sessions = np.random.binomial(1, 0.1375, 15680)