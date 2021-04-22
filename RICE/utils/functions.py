from typing import List, Sequence, Union
from functools import reduce
import operator
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

from ruleskit import Activation
from ruleskit import Rule
from ruleskit import RuleSet
from ruleskit import HyperrectangleCondition
from ruleskit.utils.rfunctions import conditional_mean


def make_condition(rule):
    """
    Evaluate all suitable rules (i.e satisfying all criteria)
    on a given feature.
    Parameters
    ----------
    rule : {rule type}
           A rule

    Return
    ------
    conditions_str : {str type}
                     A new string for the condition of the rule
    """
    conditions = rule.conditions.get_attr()
    length = len(rule)
    conditions_str = ''
    for i in range(length):
        if i > 0:
            conditions_str += ' & '

        conditions_str += conditions[0][i]
        if conditions[2][i] == conditions[3][i]:
            conditions_str += ' = '
            conditions_str += str(conditions[2][i])
        else:
            conditions_str += r' $\in$ ['
            conditions_str += str(conditions[2][i])
            conditions_str += ', '
            conditions_str += str(conditions[3][i])
            conditions_str += ']'

    return conditions_str


def make_rules(feature_name: str, feature_index: int, xs: np.ndarray, y: np.ndarray, criterion: str) -> List[Rule]:
    """
    Evaluate all suitable rules (i.e satisfying all criteria)
    on a given feature.

    Parameters
    ----------
    feature_name : {string type}
                   Name of the feature

    feature_index : {int type}
                    Columns index of the feature

    xs : {array-like or discretized matrix, shape = [n, d]}
        The training input samples after discretization.

    y : {array-like, shape = [n]}
        The normalized target values (real numbers).

    criterion : {string type}
             The method mse_function or mse_function criterion

    Return
    ------
    rules_list : {list type}
               the list of all suitable rules on the chosen feature.
    """
    rules_list = []
    xcol = xs[:, feature_index]
    xcol = np.array(xcol, dtype=np.float)
    notnan_vect = np.extract(np.isfinite(xcol), xcol)
    values = list(map(float, np.sort(list(set(notnan_vect)))))

    for bmin in values:
        j = values.index(bmin)
        for bmax in values[j:]:
            conditions = HyperrectangleCondition(features_names=[feature_name],
                                                 features_indexes=[feature_index],
                                                 bmins=[bmin],
                                                 bmaxs=[bmax])

            rule = Rule(conditions)
            rule.fit(xs, y, criterion)
            rules_list.append(rule)

    return rules_list


def fit_pair_rules(r1_r2: Sequence[Rule], xs: np.ndarray, y: np.ndarray,
                   criterion: str, length: int) -> Union[None, Rule]:
    rule = None
    r1 = r1_r2[0]
    r2 = r1_r2[1]
    if r1 is None or r2 is None:
        return rule
    if r1 == r2:
        return rule
    if len(r1) + len(r2) != length:
        raise ValueError("You arrived in length-n with two rules whose summed length is different from the total "
                         "desired length. This should have been taken care of by get_pair.")
    # noinspection PyUnresolvedReferences
    if not all([v not in r2.condition.features_names for v in r1.condition.features_names]):
        return rule
    # noinspection PyUnresolvedReferences
    rule = r1 & r2
    rule.fit(xs=xs, y=y, crit=criterion)
    return rule


def get_pair(rls, is_identical: bool = False):
    """ Get all possible pairs of elements between two lists. Will ignore Nones and same items.

    Parameters
    ---------
    rls: list
        Must contain two sublists
    is_identical: bool
        Pass True if the two elements of rls are the same lists, which will save a bit of computation time

    Example
    -------
    >>> l1 = [[1, 2], [1, 3, 4, 5, None]]
    >>> list(get_pair(l1))
    [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
    """
    if is_identical:
        for i, r1 in enumerate(rls[0]):
            for r2 in rls[1][i:]:
                if not (r1 is None or r2 is None):
                    yield r1, r2
    else:
        for r1 in rls[0]:
            for r2 in rls[1]:
                if not (r1 is None or r2 is None):
                    yield r1, r2


def calc_intersection(rule, ruleset, cov_min,
                      cov_max, X=None, low_memory=False):
    """
    Calculation of all statistics of an rules

    Parameters
    ----------
    rule : {rule type}
             An rule object

    ruleset : {ruleset type}
                 A set of rule

    cov_min : {float type such as 0 <= covmin <= 1}
              The maximal coverage of one rule

    cov_max : {float type such as 0 <= covmax <= 1}
              The maximal coverage of one rule

    X : {array-like or discretized matrix, shape = [n, d] or None}
        The training input samples after discretization.
        If low_memory is True X must not be None

    low_memory : {bool type}
                 To save activation vectors of rules

    Return
    ------
    rules_list : {list type}
                 List of rule made by intersection of rule with
                 rules from the rules set ruleset_l1.

    """
    rules_list = [rule.intersect(r, cov_min, cov_max, X, low_memory)
                  for r in ruleset]
    rules_list = list(filter(None, rules_list))  # to drop bad rules
    rules_list = list(set(rules_list))
    return rules_list


def calc_ruleset_crit(ruleset, y_train, x_train=None, method='MSE'):
    """
    Calculation of the criterion of a set of rule

    Parameters
    ----------
    ruleset : {ruleset type}
             A set of rules

    y_train : {array-like, shape = [n]}
           The normalized target values (real numbers).

    x_train : {array-like, shape = [n]}
              The normalized target values (real numbers).

    method : {string type}
             The method mse_function or mse_function criterion

    Return
    ------
    criterion : {float type}
           The value of the criteria for the method
    """
    prediction_vector, bad_cells, no_rules = ruleset.calc_pred(y_train=y_train, x_train=x_train)
    criterion = calc_criterion(prediction_vector, y_train, method)
    return criterion


def find_cluster(rs, xs, k, n_jobs):
    if len(rs) > k:
        prediction_matrix = np.array([rule.prediction * rule.get_activation(xs)
                                      for rule in rs])

        cluster_algo = KMeans(n_clusters=k, n_jobs=n_jobs)
        cluster_algo.fit(prediction_matrix)
        return cluster_algo.labels_
    else:
        return range(len(rs))


def select_candidates(rs, k):
    """
    Returns a set of candidates to increase length
    with a maximal number k
    """
    rules_list = []
    for i in range(k):
        sub_rs = rs.extract('cluster', i)
        if len(sub_rs) > 0:
            sub_rs.sort_by('var', True)
            rules_list.append(sub_rs[0])

    return RuleSet(rules_list)


def get_variables_count(rs):
    """
    Get a counter of all different features in the ruleset

    Parameters
    ----------
    rs : {ruleset type}
             A set of rules

    Return
    ------
    count : {Counter type}
            Counter of all different features in the ruleset
    """
    col_varuleset = [rule.conditions.features_names
                     for rule in rs]
    varuleset_list = reduce(operator.add, col_varuleset)
    count = Counter(varuleset_list)

    count = count.most_common()
    return count


def dist(u, v):
    """
    Compute the distance between two prediction vector

    Parameters
    ----------
    u,v : {array type}
          A predictor vector. It means a sparse array with two
          different values 0, if the rule is not active
          and the prediction is the rule is active.

    Return
    ------
    Distance between u and v
    """
    assert len(u) == len(v), \
        'The two array must have the same length'
    u = np.sign(u)
    v = np.sign(v)
    num = np.dot(u, v)
    deno = min(np.dot(u, u),
               np.dot(v, v))
    return 1 - num / deno


def mse_function(prediction_vector, y):
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)^2 $"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean squared error
    """
    assert len(prediction_vector) == len(y), \
        'The two array must have the same length'
    error_vector = prediction_vector - y
    criterion = np.nanmean(error_vector ** 2)
    return criterion


def mae_function(prediction_vector, y):
    """
    Compute the mean absolute error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} |\\hat{y}_i - y_i| $"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean absolute error
    """
    assert len(prediction_vector) == len(y), \
        'The two array must have the same length'
    error_vect = np.abs(prediction_vector - y)
    criterion = np.nanmean(error_vect)
    return criterion


def aae_function(prediction_vector, y):
    """
    Compute the mean squared error
    "$ \\dfrac{1}{n} \\Sigma_{i=1}^{n} (\\hat{y}_i - y_i)$"

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    Return
    ------
    criterion : {float type}
           the mean squared error
    """
    assert len(prediction_vector) == len(y), \
        'The two array must have the same length'
    error_vector = np.mean(np.abs(prediction_vector - y))
    median_error = np.mean(np.abs(y - np.median(y)))
    return error_vector / median_error


def calc_criterion(prediction_vector, y, method='mse'):
    """
    Compute the criteria

    Parameters
    ----------
    prediction_vector : {array type}
                A predictor vector. It means a sparse array with two
                different values ymean, if the rule is not active
                and the prediction is the rule is active.

    y : {array type}
        The real target values (real numbers)

    method : {string type}
             The method mse_function or mse_function criterion

    Return
    ------
    criterion : {float type}
           Criteria value
    """
    y_fillna = np.nan_to_num(y)

    if method == 'mse':
        criterion = mse_function(prediction_vector, y_fillna)

    elif method == 'mae':
        criterion = mae_function(prediction_vector, y_fillna)

    elif method == 'aae':
        criterion = aae_function(prediction_vector, y_fillna)

    else:
        raise 'Method %s unknown' % method

    return criterion


def significant_test(rule, ymean, sigma2, beta):
    """
    Parameters
    ----------
    rule : {Rule type}
        A rule.

    ymean : {float type}
            The mean of y.

    sigma2 : {float type}
            The noise estimator.

    beta : {float type}
            The beta factor.

    Return
    ------
    The bound for the conditional expectation to be significant
    """
    left_term = beta * abs(rule.prediction - ymean)
    right_term = np.sqrt(max(0, rule.std ** 2 - sigma2))
    return left_term > right_term


def insignificant_test(rule, sigma2, epsilon):
    return epsilon >= np.sqrt(max(0, rule.std ** 2 - sigma2))


def union_test(rule: Rule, act: Activation, gamma=0.80):
    """
    Test to know if a rule (self) and an activation vector have
    at more gamma percent of points in common
    """
    # noinspection PyProtectedMember
    rule_activation = rule._activation
    intersect_vector = rule_activation & act

    pts_inter = intersect_vector.nones
    pts_act = act.nones
    pts_rule = rule_activation.nones

    ans = (pts_inter < gamma * pts_rule) and (pts_inter < gamma * pts_act)

    return ans


def calc_coverage(vect):
    """
    Compute the coverage rate of an activation vector

    Parameters
    ----------
    vect : {array type}
           A activation vector. It means a sparse array with two
           different values 0, if the rule is not active
           and the 1 is the rule is active.

    Return
    ------
    cov : {float type}
          The coverage rate
    """
    u = np.sign(vect)
    return np.dot(u, u) / float(u.size)


def calc_prediction(activation_vector, y):
    """
    Compute the empirical conditional expectation of y
    knowing X

    Parameters
    ----------
    activation_vector : {array type}
                  A activation vector. It means a sparse array with two
                  different values 0, if the rule is not active
                  and the 1 is the rule is active.

    y : {array type}
        The target values (real numbers)

    Return
    ------
    predictions : {float type}
           The empirical conditional expectation of y
           knowing X
    """
    y_cond = np.extract(activation_vector != 0, y)
    if sum(~np.isnan(y_cond)) == 0:
        return 0
    else:
        predictions = np.nanmean(y_cond)
        return predictions


def find_bins(x, nb_bucket):
    """
    Function used to find the bins to discretize xcol in nb_bucket modalities

    Parameters
    ----------
    x : {Series type}
           Serie to discretize

    nb_bucket : {int type}
                Number of modalities

    Return
    ------
    bins : {ndarray type}
           The bins for disretization (result from numpy percentile function)
    """
    # Find the bins for nb_bucket
    q_list = np.arange(100.0 / nb_bucket, 100.0, 100.0 / nb_bucket)
    bins = np.array([np.nanpercentile(x, i) for i in q_list])

    if bins.min() != 0:
        test_bins = bins / bins.min()
    else:
        test_bins = bins

    # Test if we have same bins...
    while len(set(test_bins.round(5))) != len(bins):
        # Try to decrease the number of bucket to have unique bins
        nb_bucket -= 1
        q_list = np.arange(100.0 / nb_bucket, 100.0, 100.0 / nb_bucket)
        bins = np.array([np.nanpercentile(x, i) for i in q_list])
        if bins.min() != 0:
            test_bins = bins / bins.min()
        else:
            test_bins = bins

    return bins


def discretize(x, nb_bucket, bins=None):
    """
    Function used to have discretize xcol in nb_bucket values
    if xcol is a real series and do nothing if xcol is a string series

    Parameters
    ----------
    x : {Series type}
           Series to discretize

    nb_bucket : {int type}
                Number of modalities

    bins : {ndarray type}, optional, default None
           If you have already calculate the bins for xcol

    Return
    ------
    x_discretized : {Series type}
                       The discretization of xcol
    """
    if np.issubdtype(x.dtype, np.floating):
        # extraction of the list of xcol values
        notnan_vector = np.extract(np.isfinite(x), x)
        nan_index = ~np.isfinite(x)
        # Test if xcol have more than nb_bucket different values
        if len(set(notnan_vector)) >= nb_bucket or bins is not None:
            if bins is None:
                bins = find_bins(x, nb_bucket)
            # discretization of the xcol with bins
            x_discrete = np.digitize(x, bins=bins)
            x_discrete = np.array(x_discrete, dtype='float')
            if sum(nan_index) > 0:
                x_discrete[nan_index] = np.nan

            return x_discrete

        return x
    else:
        return x


def predict(rs: RuleSet, xs: np.ndarray, y_train: np.ndarray) -> (np.ndarray, np.ndarray):
    max_func = np.vectorize(max)
    significant_rules = list(filter(lambda rule: rule.significant, rs))
    insignificant_rules = list(filter(lambda rule: rule.significant is False, rs))

    if len(significant_rules) > 0:
        # noinspection PyProtectedMember
        significant_union = reduce(operator.add, [rule._activation for rule in significant_rules]).raw

        significant_act_matrix = [rule.activation for rule in significant_rules]
        significant_act_matrix = np.array(significant_act_matrix)

        significant_pred_matrix = [rule.evaluate(xs).raw for rule in significant_rules]
        significant_pred_matrix = np.array(significant_pred_matrix).T

        no_activation_matrix = np.logical_not(significant_pred_matrix)

        nb_rules_active = significant_pred_matrix.sum(axis=1)
        nb_rules_active[nb_rules_active == 0] = -1  # If no rule is activated

        # Activation of the intersection of all NOT activated rules at each row
        no_activation_vector = np.dot(no_activation_matrix, significant_act_matrix)
        no_activation_vector = np.array(no_activation_vector, dtype='int')

        # Activation of the intersection of all activated rules at each row
        dot_activation = np.dot(significant_pred_matrix, significant_act_matrix)
        dot_activation = np.array([np.equal(act, nb_rules) for act, nb_rules in
                                   zip(dot_activation, nb_rules_active)], dtype='int')

        # Calculation of the binary vector for cells of the partition et each row
        significant_cells = ((dot_activation - no_activation_vector) > 0)
        no_prediction_points = (significant_cells.sum(axis=1) == 0) & (significant_pred_matrix.sum(axis=1) != 0)

    else:
        significant_cells = np.zeros(shape=(xs.shape[0], len(y_train)))
        significant_union = np.zeros(len(y_train))
        no_prediction_points = np.zeros(xs.shape[0])

    if len(insignificant_rules) > 0:
        # Activation of all rules in the learning set
        insignificant_act_matrix = [rule.activation for rule in insignificant_rules]
        insignificant_act_matrix = np.array(insignificant_act_matrix)
        insignificant_act_matrix -= significant_union
        insignificant_act_matrix = max_func(insignificant_act_matrix, 0)

        insignificant_pred_matrix = [rule.evaluate(xs).raw for rule in insignificant_rules]
        insignificant_pred_matrix = np.array(insignificant_pred_matrix).T

        no_activation_matrix = np.logical_not(insignificant_pred_matrix)

        nb_rules_active = insignificant_pred_matrix.sum(axis=1)
        nb_rules_active[nb_rules_active == 0] = -1  # If no rule is activated

        # Activation of the intersection of all NOT activated rules at each row
        no_activation_vector = np.dot(no_activation_matrix, insignificant_act_matrix)
        no_activation_vector = np.array(no_activation_vector, dtype='int')

        # Activation of the intersection of all activated rules at each row
        dot_activation = np.dot(insignificant_pred_matrix, insignificant_act_matrix)
        dot_activation = np.array([np.equal(act, nb_rules) for act, nb_rules in
                                   zip(dot_activation, nb_rules_active)], dtype='int')

        # Calculation of the binary vector for cells of the partition et each row
        insignificant_cells = ((dot_activation - no_activation_vector) > 0)
    else:
        insignificant_cells = np.zeros(shape=(xs.shape[0], len(y_train)))

    # Calculation of the No-rule prediction.
    no_rule_cell = np.ones(len(y_train)) - significant_union
    no_rule_prediction = conditional_mean(no_rule_cell, y_train)

    # Calculation of the conditional expectation in each cell
    cells = insignificant_cells ^ significant_cells
    prediction_vector = [conditional_mean(act, y_train) if sum(act) > 0 else 0.0 for act in cells]
    prediction_vector = np.array(prediction_vector)
    prediction_vector[no_prediction_points] = np.nan
    prediction_vector[prediction_vector == 0] = no_rule_prediction

    return np.array(prediction_vector), no_prediction_points
