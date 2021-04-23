from typing import Optional
import numpy as np
from ruleskit import Rule
from ruleskit import Condition
from ruleskit import Activation


class RiceRule(Rule):
    def __init__(self, condition: Optional[Condition] = None, activation: Optional[Activation] = None):
        super(RiceRule, self).__init__(condition, activation)
        self.out = False

    def fit(self, xs: np.ndarray, y: np.ndarray, crit: str = "mse", cov_min: float = 0.1):
        """Computes activation, prediction, std and criteria of the rule for a given xs and y."""
        self.calc_activation(xs)  # returns Activation
        if self.coverage < cov_min:
            self.out = True
        else:
            self.calc_prediction(y)
            self.calc_std(y)
            prediction_vector = self.prediction * self.activation
            self.calc_criterion(prediction_vector, y, crit)
