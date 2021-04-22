# -*- coding: utf-8 -*-
"""
Created on 22 sept. 2016
@author: VMargot
"""
from typing import List, Union
from functools import reduce
import operator
from .utils import functions as f
import numpy as np
from joblib import Parallel, delayed
from ruleskit import RuleSet
from ruleskit import Rule
from ruleskit.utils import rfunctions
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_array


class RICE:
    """
    ...
    """
    def __init__(self, **parameters):
        """

        Parameters
        ----------
        alpha : {float type such as 0 < th < 1/4} default 1/5
                The main parameter

        nb_bucket : {int type} default max(3, n^1/d) with n the number of row
                    and d the number of features
                    Choose the number a bucket for the discretization

        l_max : {int type} default d
                 Choose the maximal length of one rule

        gamma : {float type such as 0 <= gamma <= 1} default 1
                Choose the maximal intersection rate begin a rule and
                a current selected ruleset

        k : {int type} default 500
            The maximal number of candidate to increase length

        nb_jobs : {int type} default number of core -2
                  Select the number of lU used
        """
        self.selected_rs = RuleSet([])
        self.rs = RuleSet([])
        self.fitted = False
        self.nb_jobs = -2
        self.features_names = None

        self._bins = dict()
        self._alpha = 1. / 2 - 1. / 100
        self._beta = None
        self._epsilon = None
        self._sigma2 = None
        self._gamma = 0.95
        self._covmin = None
        self._covmax = 1
        self._nb_bucket = None
        self._criterion = 'auto'
        self._lmax = None
        self._xtrain = None
        self._ytrain = None

        for arg, val in parameters.items():
            setattr(self, arg, val)

    def __str__(self):
        learning = 'Learning'
        return learning

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def lmax(self) -> int:
        return self._lmax

    @property
    def covmin(self) -> float:
        return self._covmin

    @property
    def nb_bucket(self) -> int:
        return self._nb_bucket

    @property
    def bins(self) -> dict:
        return self._bins

    @property
    def criterion(self) -> str:
        return self._criterion

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def sigma2(self) -> float:
        return self._sigma2

    @property
    def xtrain(self) -> np.ndarray:
        return self._xtrain

    @property
    def ytrain(self) -> np.ndarray:
        return self._ytrain

    @alpha.setter
    def alpha(self, values: float):
        self._alpha = values

    @beta.setter
    def beta(self, values: float):
        self._beta = values

    @epsilon.setter
    def epsilon(self, values: float):
        self._epsilon = values

    @lmax.setter
    def lmax(self, values: int):
        self._lmax = values

    @covmin.setter
    def covmin(self, values: float):
        self._covmin = values

    @nb_bucket.setter
    def nb_bucket(self, values: int):
        self._nb_bucket = values

    @bins.setter
    def bins(self, values: dict):
        self._bins = values

    @gamma.setter
    def gamma(self, values: float):
        self._gamma = values

    @sigma2.setter
    def sigma2(self, values: float):
        self._sigma2 = values

    @criterion.setter
    def criterion(self, values: str):
        self._criterion = values

    @xtrain.setter
    def xtrain(self, values: np.ndarray):
        self._xtrain = values

    @ytrain.setter
    def ytrain(self, values: np.ndarray):
        self._ytrain = values

    def fit(self, xs: np.ndarray, y: np.ndarray, features_names: List[str] = None):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        xs : {array-like or sparse matrix, shape = [n, d]}
            The training input samples.

        y : {array-like, shape = [n]}
            The target values (real numbers).

        features_names : {list}, optional
                        Name of each features
        """
        # Check type for data
        # xs = check_array(xs, dtype=np.ndarray, force_all_finite=False)
        # y = check_array(y, dtype=np.ndarray, ensure_2d=False, force_all_finite=False)

        # Creation of data-driven parameters
        if self.beta is None:
            self.beta = 1. / pow(xs.shape[0], 1. / 4 - self.alpha / 2.)

        if self.epsilon is None:
            self.epsilon = self.beta * np.std(y)

        if self.covmin is None:
            self.covmin = 1. / pow(xs.shape[0], self.alpha)

        if self.nb_bucket is None:
            self.nb_bucket = min(max(5, int(np.sqrt(pow(xs.shape[0], 1. / xs.shape[1])))),
                                 xs.shape[0])

        if self.criterion == 'auto':
            if len(set(y)) > 2:
                # Binary classification case
                self.criterion = 'mse'
            else:
                # Regression case
                self.criterion = 'mae'

        features_indexes = range(xs.shape[1])
        if features_names is None:
            features_names = ['X' + str(i) for i in features_indexes]
        self.features_names = features_names

        if self.lmax is None:
            self.lmax = len(features_names)

        # Turn the matrix X in a discrete matrix
        discrete_xs = self.discretize(xs)

        # --------------
        # DESIGNING PART
        # --------------
        print('----- Design ------')
        rs = self.design_rules(discrete_xs, y, features_indexes, features_names)  # works in columns not in lines
        self.rs = rs
        # --------------
        # SELECTION PART
        # --------------
        print('----- Selection ------')
        selected_rs = self.select_rules(rs, 0, np.mean(y))
        self.selected_rs = selected_rs

        print('----- Fitting is over ------')
        self.fitted = True
        self.xtrain = discrete_xs
        self.ytrain = y

    def design_rules(self, discrete_xs: np.ndarray, y: np.ndarray,
                     features_indexes: Union[range, List[int]], features_names: List[str]) -> RuleSet:
        """
        Find all rules for all length <= l
        then selects the best subset by minimization
        of the empirical risk
        """
        rules_list = self.calc_length_1(discrete_xs, y, features_indexes, features_names)
        rules_list = list(filter(lambda rule: rule.coverage >= self.covmin, rules_list))

        if len(rules_list) > 0:
            for k in range(2, self.lmax + 1):
                print('Designing of rules of length %s' % str(k))
                rules_length_k = self.calc_length_l(rules_list, discrete_xs, y, k)
                if len(rules_length_k) > 0:
                    rules_length_k = list(filter(lambda rule: rule.coverage >= self.covmin, rules_length_k))
                    rules_list += rules_length_k
                else:
                    print('No rule for length %s' % str(k))

            rules_list = sorted(rules_list, key=lambda x: x.criterion, reverse=False)
            return RuleSet(rules_list)
        else:
            print('No rule found !')
            return RuleSet([])

    def calc_length_1(self, discrete_xs: np.ndarray, y: np.ndarray,
                      features_indexes: List[int], features_names: List[str]) -> List[Rule]:
        """
        Compute all rules of length one and keep the best.
        """
        indices = zip(features_names, features_indexes)
        jobs = min(len(features_names), self.nb_jobs)
        if jobs == 1:
            rules_list = [f.make_rules(c_name, c_idx, discrete_xs, y, self.criterion) for c_name, c_idx in indices]
        else:
            rules_list = Parallel(n_jobs=jobs, backend="multiprocessing")(
                delayed(f.make_rules)(c_name, c_idx, discrete_xs, y, self.criterion) for c_name, c_idx in indices)

        rules_list = reduce(operator.add, rules_list)
        rules_list = sorted(rules_list, key=lambda x: x.criterion, reverse=False)

        return rules_list

    def calc_length_l(self, rules: List[Rule], discrete_xs: np.ndarray, y: np.ndarray, length: int) -> List[Rule]:
        """
        Returns a ruleset of rules with a given length.
        """
        nb_jobs = self.nb_jobs
        criterion = self.criterion
        rules_pairs = f.get_pair([list(filter(lambda rule: len(rule) == 1, rules)),
                                  list(filter(lambda rule: len(rule) == length - 1, rules))])
        if nb_jobs == 1:
            rules_list = [f.fit_pair_rules(r1_r2, discrete_xs, y, criterion, length) for r1_r2 in rules_pairs]
        else:
            rules_list = Parallel(n_jobs=nb_jobs, backend="multiprocessing")(
                delayed(f.fit_pair_rules)(r1_r2, discrete_xs, y, criterion, length) for r1_r2 in rules_pairs)
        rules_list = list(filter(lambda rule: rule is not None, rules_list))
        rules_list = sorted(rules_list, key=lambda x: x.criterion, reverse=False)

        return rules_list

    def select_rules(self, rs: RuleSet, length: int, ymean: float):
        """
        Returns a subset of a given ruleset.
        This subset minimizes the empirical contrast on the learning set
        """
        beta = self.beta
        epsilon = self.epsilon
        selected_rs = RuleSet([])

        if length > 0:
            sub_rs = RuleSet(list(filter(lambda rule: len(rule) == length, rs)))
        else:
            sub_rs = rs

        print('Number of rules: %s' % str(len(sub_rs)))

        if self.sigma2 is None:
            self.sigma2 = min([rule.std ** 2 for rule in sub_rs])

        # Selection of significant rules
        significant_rules = list(filter(lambda rule: f.significant_test(rule, ymean, self.sigma2, beta), sub_rs))
        if len(significant_rules) > 0:
            [setattr(rule, 'significant', True) for rule in significant_rules]
            print('Number of rules after significant test: %s'
                  % str(len(significant_rules)))
            sorted_significant = sorted(significant_rules, key=lambda x: x.coverage, reverse=True)
            significant_rs = RuleSet(list(sorted_significant))

            rg_add, selected_rs = self.select(significant_rs)
            print('Number of selected significant rules: %s' % str(rg_add))

        else:
            print('No significant rules selected!')

        # Add insignificant rules to the current selection set of rules
        if selected_rs is None or selected_rs.calc_coverage_rate() < 1:
            insignificant_list = filter(lambda rule: f.insignificant_test(rule, self.sigma2,  epsilon), sub_rs)
            insignificant_list = list(filter(lambda rule: rule not in significant_rules,
                                             insignificant_list))
            if len(list(insignificant_list)) > 0:
                [setattr(rule, 'significant', False) for rule in insignificant_list]
                print('Number rules after insignificant test: %s'
                      % str(len(insignificant_list)))

                insignificant_list = sorted(insignificant_list, key=lambda x: x.std, reverse=False)
                insignificant_rs = RuleSet(list(insignificant_list))
                rg_add, selected_rs = self.select(insignificant_rs, selected_rs)
                print('Number insignificant rules added: %s' % str(rg_add))
            else:
                print('No insignificant rule added.')
        else:
            print('Covering is completed. No insignificant rule added.')

        # Add rule to have a covering
        coverage_rate = selected_rs.calc_coverage_rate()
        if coverage_rate < 1:
            print('Warning: Covering is not completed!', coverage_rate)
            # neg_rule, pos_rule = add_no_rule(selected_rs, x_train, y_train)
            # features_name = self.get_param('features_name')
            #
            # if neg_rule is not None:
            #     id_feature = neg_rule.conditions.get_param('features_index')
            #     rule_features = list(itemgetter(*id_feature)(features_name))
            #     neg_rule.conditions.set_params(features_name=rule_features)
            #     neg_rule.calc_stats(y=y_train, x=x_train, cov_min=0.0, cov_max=1.0)
            #     print('Add negative no-rule  %s.' % str(neg_rule))
            #     selected_rs.append(neg_rule)
            #
            # if pos_rule is not None:
            #     id_feature = pos_rule.conditions.get_param('features_index')
            #     rule_features = list(itemgetter(*id_feature)(features_name))
            #     pos_rule.conditions.set_params(features_name=rule_features)
            #     pos_rule.calc_stats(y=y_train, x=x_train, cov_min=0.0, cov_max=1.0)
            #     print('Add positive no-rule  %s.' % str(pos_rule))
            #     selected_rs.append(pos_rule)
        else:
            print('Covering is completed.')

        return selected_rs

    def select(self, rs: RuleSet, selected_rs: RuleSet = None) -> (int, RuleSet):
        gamma = self.gamma
        i = 0
        rg_add = 0

        if selected_rs is None:
            selected_rs = RuleSet(rs[:1])
            rs = RuleSet(rs[1:])
            rg_add += 1

        nb_rules = len(rs)
        # old_criterion = calc_ruleset_crit(selected_rs, y_train, x_train, calcmethod)
        # crit_evo.append(old_criterion)

        while selected_rs.calc_coverage_rate() < 1 and i < nb_rules:
            new_rules = rs[i]

            # noinspection PyProtectedMember
            utests = [f.union_test(new_rules, rule._activation, gamma) for rule in selected_rs]
            if all(utests) and f.union_test(new_rules, selected_rs.get_activation(), gamma):
                selected_rs += new_rules
                # old_criterion = new_criterion
                rg_add += 1

            # crit_evo.append(old_criterion)
            i += 1

        # self.set_params(critlist=crit_evo)
        return rg_add, selected_rs

    def predict(self, xs, check_input=True) -> (np.ndarray, np.ndarray):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        application of the selected ruleset on X.

        Parameters
        ----------
        xs : {array type or sparse matrix of shape = [n_samples, n_features]}
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a spares matrix is provided, it will be
            converted into a spares ``csr_matrix``.

        check_input : bool type

        Returns
        -------
        y : {array type of shape = [n_samples]}
            The predicted values.
        """
        y_train = self.ytrain
        xs = self.validate_xs_predict(xs, check_input)
        discrete_xs = self.discretize(xs)
        selected_rs = self.selected_rs

        prediction_vector, no_predictions = f.predict(selected_rs, discrete_xs, y_train)
        return prediction_vector, no_predictions

    def score(self, x, y, sample_weight=None):
        """
        Returns the coefficient of determination R^2 of the prediction
        if y is continuous. Else if y in {0,1} then Returns the mean
        accuracy on the given test data and labels {0,1}.

        Parameters
        ----------
        x : {array type or sparse matrix of shape = [n_samples, n_features]}
            Test samples.

        y : {array type of shape = [n_samples]}
            True values for y.

        sample_weight : {array type of shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y in R.
            or
        score : float
            Mean accuracy of self.predict(X) wrt. y in {0,1}
        """
        prediction_vector, no_predictions = self.predict(x)
        print(f'There are {sum(no_predictions)} observations without prediction.')
        prediction_vector = np.nan_to_num(prediction_vector)

        nan_val = np.argwhere(np.isnan(y))
        if len(nan_val) > 0:
            prediction_vector = np.delete(prediction_vector, nan_val)
            y = np.delete(y, nan_val)

        if len(set(y)) == 2:
            th_val = (min(y) + max(y)) / 2.0
            prediction_vector = list(map(lambda p: min(y) if p < th_val else max(y),
                                         prediction_vector))
            return accuracy_score(y, prediction_vector)
        else:
            return r2_score(y, prediction_vector, sample_weight=sample_weight,
                            multioutput='variance_weighted')

    """------   Data functions   -----"""
    def validate_xs_predict(self, xs, check_input):
        """
        Validate X whenever one tries to predict, apply, predict_proba
        """
        if self.fitted is False:
            raise AttributeError("Estimator not fitted, call 'fit' before exploiting the model.")

        if check_input:
            xs = check_array(xs, dtype=None, force_all_finite=False)

            n_features = xs.shape[1]
            input_features = self.features_names
            if len(input_features) != n_features:
                raise ValueError("Number of features of the model must "
                                 "match the input. Model n_features is %s and "
                                 "input n_features is %s "
                                 % (input_features, n_features))

        return xs

    def discretize(self, x):
        """
        Used to have discrete values for each series
        to avoid float

        Parameters
        ----------
        x : {array, matrix type}, shape=[n_samples, n_features]
            Features matrix

        Return
        -------
        col : {array, matrix type}, shape=[n_samples, n_features]
              Features matrix with each features values discretized
              in nb_bucket values
        """
        nb_col = x.shape[1]
        nb_bucket = self.nb_bucket
        bins_dict = self.bins
        features_names = self.features_names

        x_mat = []
        for i in range(nb_col):
            xcol = x[:, i]
            try:
                xcol = np.array(xcol.flat, dtype=np.float)
            except ValueError:
                xcol = np.array(xcol.flat, dtype=np.str)

            var_name = features_names[i]

            if np.issubdtype(xcol.dtype, np.floating):
                if var_name not in bins_dict:
                    if len(set(xcol)) >= nb_bucket:
                        bins = f.find_bins(xcol, nb_bucket)
                        discrete_column = f.discretize(xcol, nb_bucket, bins)
                        bins_dict[var_name] = bins
                    else:
                        discrete_column = xcol
                else:
                    bins = bins_dict[var_name]
                    discrete_column = f.discretize(xcol, nb_bucket, bins)
            else:
                discrete_column = xcol

            x_mat.append(discrete_column)

        return np.array(x_mat).T

    # def plot_rules(self, var1, var2, length=None,
    #                col_pos='red', col_neg='blue'):
    #     """
    #     Plot the rectangle activation zone of rules in a 2D plot
    #     the color is corresponding to the intensity of the prediction
    #
    #     Parameters
    #     ----------
    #     var1 : {string type}
    #            Name of the first variable
    #
    #     var2 : {string type}
    #            Name of the second variable
    #
    #     length : {int type}, optional
    #              Option to plot only the length 1 or length 2 rules
    #
    #     col_pos : {string type}, optional,
    #               Name of the color of the zone of positive rules
    #
    #     col_neg : {string type}, optional
    #               Name of the color of the zone of negative rules
    #
    #     -------
    #     Draw the graphic
    #     """
    #     selected_rs = self.get_param('selected_rs')
    #     nb_bucket = self.get_param('nb_bucket')
    #
    #     if length is not None:
    #         sub_ruleset = selected_rs.extract_length(length)
    #     else:
    #         sub_ruleset = selected_rs
    #
    #     plt.plot()
    #
    #     for rule in sub_ruleset:
    #         rule_condition = rule.conditions
    #
    #         var = rule_condition.get_param('features_index')
    #         bmin = rule_condition.get_param('bmin')
    #         bmax = rule_condition.get_param('bmax')
    #         length_rule = rule.get_param('length')
    #
    #         if rule.get_param('pred') > 0:
    #             hatch = '/'
    #             facecolor = col_pos
    #             alpha = min(1, abs(rule.get_param('pred')) / 2.0)
    #         else:
    #             hatch = '\\'
    #             facecolor = col_neg
    #             alpha = min(1, abs(rule.get_param('pred')) / 2.0)
    #
    #         if length_rule == 1:
    #             if var[0] == var1:
    #                 p = patches.Rectangle((bmin[0], 0),  # origin
    #                                       (bmax[0] - bmin[0]) + 0.99,  # width
    #                                       nb_bucket,  # height
    #                                       hatch=hatch, facecolor=facecolor,
    #                                       alpha=alpha)
    #                 plt.gca().add_patch(p)
    #
    #             elif var[0] == var2:
    #                 p = patches.Rectangle((0, bmin[0]),
    #                                       nb_bucket,
    #                                       (bmax[0] - bmin[0]) + 0.99,
    #                                       hatch=hatch, facecolor=facecolor,
    #                                       alpha=alpha)
    #                 plt.gca().add_patch(p)
    #
    #         elif length_rule == 2:
    #             if var[0] == var1 and var[1] == var2:
    #                 p = patches.Rectangle((bmin[0], bmin[1]),
    #                                       (bmax[0] - bmin[0]) + 0.99,
    #                                       (bmax[1] - bmin[1]) + 0.99,
    #                                       hatch=hatch, facecolor=facecolor,
    #                                       alpha=alpha)
    #                 plt.gca().add_patch(p)
    #
    #             elif var[1] == var1 and var[0] == var2:
    #                 p = patches.Rectangle((bmin[1], bmin[0]),
    #                                       (bmax[1] - bmin[1]) + 0.99,
    #                                       (bmax[0] - bmin[0]) + 0.99,
    #                                       hatch=hatch, facecolor=facecolor,
    #                                       alpha=alpha)
    #                 plt.gca().add_patch(p)
    #
    #     if length is None:
    #         plt.gca().set_title('rules activations')
    #     else:
    #         plt.gca().set_title('rules l%s activations' % str(length))
    #
    #     plt.gca().axis([-0.1, nb_bucket + 0.1, -0.1, nb_bucket + 0.1])
    #
    # def plot_pred(self, x, y, var1, var2, cmap=None,
    #               vmin=None, vmax=None, add_points=True,
    #               add_score=False):
    #     """
    #     Plot the prediction zone of rules in a 2D plot
    #
    #     Parameters
    #     ----------
    #     x : {array-like, sparse matrix}, shape=[n_samples, n_features]
    #         Features matrix, where n_samples in the number of samples and
    #         n_features is the number of features.
    #
    #     y : {array-like}, shape=[n_samples]
    #         Target vector relative to X
    #
    #     var1 : {int type}
    #            Number of the column of the first variable
    #
    #     var2 : {int type}
    #            Number of the column of the second variable
    #
    #     cmap : {colormap object}, optional
    #            Colormap used for the graphic
    #
    #     vmax, vmin : {float type}, optional
    #                  Parameter of the range of the colorbar
    #
    #     add_points: {boolean type}, optional
    #                 Option to add the discrete scatter of y
    #
    #     add_score : {boolean type}, optional
    #                 Option to add the score on the graphic
    #
    #     -------
    #     Draw the graphic
    #     """
    #     nb_bucket = self.get_param('nb_bucket')
    #     x_discretized = self.discretize(x)
    #     selected_rs = self.get_param('selected_rs')
    #     y_train = self.get_param('y')
    #     ymean = self.get_param('ymean')
    #     ystd = self.get_param('ystd')
    #
    #     x1 = x_discretized[:, var1]
    #     x2 = x_discretized[:, var2]
    #
    #     xx, yy = np.meshgrid(range(nb_bucket),
    #                          range(nb_bucket))
    #
    #     if cmap is None:
    #         cmap = plt.cm.get_cmap('coolwarm')
    #
    #     z = selected_rs.predict(y_train, np.c_[np.round(xx.ravel()),
    #                                            np.round(yy.ravel())],
    #                             ymean, ystd)
    #
    #     if vmin is None:
    #         vmin = min(z)
    #     if vmax is None:
    #         vmax = max(z)
    #
    #     z = z.reshape(xx.shape)
    #
    #     plt.contourf(xx, yy, z, cmap=cmap, alpha=.8, vmax=vmax, vmin=vmin)
    #
    #     if add_points:
    #         area = map(lambda b:
    #                    map(lambda a:
    #                        np.extract(np.logical_and(x1 == a, x2 == b),
    #                                   y).mean(), range(nb_bucket)),
    #                    range(nb_bucket))
    #         area = list(area)
    #
    #         area_len = map(lambda b:
    #                        map(lambda a:
    #                            len(np.extract(np.logical_and(x1 == a, x2 == b),
    #                                           y)) * 10, range(nb_bucket)),
    #                        range(nb_bucket))
    #         area_len = list(area_len)
    #
    #         plt.scatter(xx, yy, c=area, s=area_len, alpha=1.0,
    #                     cmap=cmap, vmax=vmax, vmin=vmin)
    #
    #     plt.title('RIPE prediction')
    #
    #     if add_score:
    #         score = self.score(x, y)
    #         plt.text(nb_bucket - .70, .08, ('%.2f' % str(score)).lstrip('0'),
    #                  size=20, horizontalalignment='right')
    #
    #     plt.axis([-0.01, nb_bucket - 0.99, -0.01, nb_bucket - 0.99])
    #     plt.colorbar()
    #
    # def plot_counter_variables(self):
    #     """
    #     Function plots a graphical counter of variables used in rules.
    #     """
    #     rs = self.get_param('selected_rs')
    #     f = rs.plot_counter_variables()
    #
    #     return f
    #
    # def plot_counter(self):
    #     """
    #     Function plots a graphical counter of variables used in rules by modality.
    #     """
    #     nb_bucket = self.get_param('nb_bucket')
    #     y_labels, counter = self.make_count_matrix(return_vars=True)
    #
    #     x_labels = list(map(lambda i: str(i), range(nb_bucket)))
    #
    #     f = plt.figure()
    #     ax = plt.subplot()
    #
    #     g = sns.heatmap(counter, xticklabels=x_labels, yticklabels=y_labels,
    #                     cmap='Reds', linewidths=.05, ax=ax, center=0.0)
    #     g.xaxis.tick_top()
    #     plt.yticks(rotation=0)
    #
    #     return f
    #
    # def plot_dist(self, x=None):
    #     """
    #     Function plots a graphical correlation of rules.
    #     """
    #     rs = self.get_param('selected_rs')
    #     if x is None and self.get_param('low_memory'):
    #         x = self.get_param('X')
    #
    #     f = rs.plot_dist(x=x)
    #
    #     return f
    #
    # def plot_intensity(self):
    #     """
    #     Function plots a graphical counter of variables used in rules.
    #     """
    #     y_labels, counter = self.make_count_matrix(return_vars=True)
    #     intensity = self.make_count_matrix(add_pred=True)
    #
    #     nb_bucket = self.get_param('nb_bucket')
    #     x_labels = [str(i) for i in range(nb_bucket)]
    #
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         val = np.divide(intensity, counter)
    #
    #     val[np.isneginf(val)] = np.nan
    #     val = np.nan_to_num(val)
    #
    #     f = plt.figure()
    #     ax = plt.subplot()
    #
    #     g = sns.heatmap(val, xticklabels=x_labels, yticklabels=y_labels,
    #                     cmap='bwr', linewidths=.05, ax=ax, center=0.0)
    #     g.xaxis.tick_top()
    #     plt.yticks(rotation=0)
    #
    #     return f
    #
    # def make_count_matrix(self, add_pred=False, return_vars=False):
    #     """
    #     Return a count matrix of each variable in each modality
    #     """
    #     ruleset = self.get_param('selected_rs')
    #     nb_bucket = self.get_param('nb_bucket')
    #
    #     counter = get_variables_count(ruleset)
    #
    #     vars_list = [item[0] for item in counter]
    #
    #     count_mat = np.zeros((nb_bucket, len(vars_list)))
    #     str_id = []
    #
    #     for rule in ruleset:
    #         cd = rule.conditions
    #         var_name = cd.get_param('features_name')
    #         bmin = cd.get_param('bmin')
    #         bmax = cd.get_param('bmax')
    #
    #         for j in range(len(var_name)):
    #             if type(bmin[j]) != str:
    #                 for b in range(int(bmin[j]), int(bmax[j]) + 1):
    #                     var_id = vars_list.index(var_name[j])
    #                     if add_pred:
    #                         count_mat[b, var_id] += rule.get_param('pred')
    #                     else:
    #                         count_mat[b, var_id] += 1
    #             else:
    #                 str_id += [vars_list.index(var_name[j])]
    #
    #     vars_list = [i for j, i in enumerate(vars_list) if j not in str_id]
    #     count_mat = np.delete(count_mat.T, str_id, 0)
    #
    #     if return_vars:
    #         return vars_list, count_mat
    #     else:
    #         return count_mat
    #
    # def make_selected_df(self):
    #     """
    #     Returns
    #     -------
    #     selected_df : {DataFrame type}
    #                   DataFrame of selected RuleSet for presentation
    #     """
    #     selected_rs = self.selected_rs
    #     selected_df = selected_rs.make_selected_df()
    #     return selected_df
