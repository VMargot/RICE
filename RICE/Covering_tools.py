import numpy as np
import tqdm
import copy
import math
from operator import itemgetter
from sklearn.tree import _tree
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches

import RICE


def inter(rs):
    return sum(map(lambda r: r.length, rs))


def extract_rules_from_tree(tree, features, xmin, xmax):
    dt = tree.tree_
    
    def visitor(node, depth, cond=None, rule_list=None):
        if rule_list is None:
            rule_list = []
        if dt.feature[node] != _tree.TREE_UNDEFINED:
            # If
            new_cond = RICE.RuleConditions([features[dt.feature[node]]],
                                           [dt.feature[node]],
                                           bmin=[xmin[dt.feature[node]]],
                                           bmax=[dt.threshold[node]],
                                           xmin=[xmin[dt.feature[node]]],
                                           xmax=[xmax[dt.feature[node]]])
            if cond is not None:
                if dt.feature[node] not in cond.features_index:
                    conditions_list = list(map(lambda c1, c2: c1 + c2, cond.get_attr(),
                                               new_cond.get_attr()))
                    
                    new_cond = RICE.RuleConditions(features_name=conditions_list[0],
                                                   features_index=conditions_list[1],
                                                   bmin=conditions_list[2],
                                                   bmax=conditions_list[3],
                                                   xmax=conditions_list[5],
                                                   xmin=conditions_list[4])
                else:
                    new_bmax = dt.threshold[node]
                    new_cond = copy.deepcopy(cond)
                    place = cond.features_index.index(dt.feature[node])
                    new_cond.bmax[place] = min(new_bmax, new_cond.bmax[place])
            
            # print (RICE.Rule(new_cond))
            new_rg = RICE.Rule(copy.deepcopy(new_cond))
            rule_list.append(new_rg)
            
            rule_list = visitor(dt.children_left[node], depth + 1,
                                new_cond, rule_list)
            
            # Else
            new_cond = RICE.RuleConditions([features[dt.feature[node]]],
                                           [dt.feature[node]],
                                           bmin=[dt.threshold[node]],
                                           bmax=[xmax[dt.feature[node]]],
                                           xmin=[xmin[dt.feature[node]]],
                                           xmax=[xmax[dt.feature[node]]])
            if cond is not None:
                if dt.feature[node] not in cond.features_index:
                    conditions_list = list(map(lambda c1, c2: c1 + c2, cond.get_attr(),
                                               new_cond.get_attr()))
                    new_cond = RICE.RuleConditions(features_name=conditions_list[0],
                                                   features_index=conditions_list[1],
                                                   bmin=conditions_list[2],
                                                   bmax=conditions_list[3],
                                                   xmax=conditions_list[5],
                                                   xmin=conditions_list[4])
                else:
                    new_bmin = dt.threshold[node]
                    new_bmax = xmax[dt.feature[node]]
                    new_cond = copy.deepcopy(cond)
                    place = new_cond.features_index.index(dt.feature[node])
                    new_cond.bmin[place] = max(new_bmin, new_cond.bmin[place])
                    new_cond.bmax[place] = max(new_bmax, new_cond.bmax[place])
            
            # print (RICE.Rule(new_cond))
            new_rg = RICE.Rule(copy.deepcopy(new_cond))
            rule_list.append(new_rg)
            
            rule_list = visitor(dt.children_right[node], depth + 1, new_cond, rule_list)
        
        return rule_list
    
    rule_list = visitor(0, 1)
    return rule_list


def extract_rules_rulefit(rules, features, bmin_list, bmax_list):
    rule_list = []
    
    for rule in rules['rule'].values:
        if '&' in rule:
            rule_split = rule.split(' & ')
        else:
            rule_split = [rule]
        
        features_name = []
        features_index = []
        bmin = []
        bmax = []
        xmax = []
        xmin = []
        
        for sub_rule in rule_split:
            sub_rule = sub_rule.replace('=', '')
            
            if '>' in sub_rule:
                sub_rule = sub_rule.split(' > ')
                feat_id = sub_rule[0].split('_')[-1]
                feat_id = int(feat_id)
                features_name += [features[feat_id]]
                features_index += [feat_id]
                bmin += [float(sub_rule[-1])]
                bmax += [bmax_list[feat_id]]
            else:
                sub_rule = sub_rule.split(' < ')
                feat_id = sub_rule[0].split('_')[-1]
                feat_id = int(feat_id)
                features_name += [features[feat_id]]
                features_index += [feat_id]
                bmax += [float(sub_rule[-1])]
                bmin += [bmin_list[feat_id]]
            
            xmax += [bmax_list[feat_id]]
            xmin += [bmin_list[feat_id]]
        
        new_cond = RICE.RuleConditions(features_name=features_name,
                                       features_index=features_index,
                                       bmin=bmin, bmax=bmax,
                                       xmin=xmin, xmax=xmax)
        new_rg = RICE.Rule(copy.deepcopy(new_cond))
        rule_list.append(new_rg)
    
    return rule_list


def select_rs(rs, gamma=1.0, selected_rs=None):
    """
    Returns a subset of a given rs. This subset is seeking by
    minimization/maximization of the criterion on the training set
    """
    # Then optimization
    if selected_rs is None or len(selected_rs) == 0:
        selected_rs = RICE.RuleSet(rs[:1])
        id_rule = 1
    else:
        id_rule = 0
    
    nb_rules = len(rs)
    
    # for i in tqdm.tqdm(range(id_rule, nb_rules), desc='Selection'):
    for i in range(id_rule, nb_rules):
        rs_copy = copy.deepcopy(selected_rs)
        new_rules = rs[i]
        
        utest = [new_rules.union_test(rule.get_activation(),
                                      gamma)
                 for rule in rs_copy]
        
        if all(utest) and new_rules.union_test(selected_rs.calc_activation(),
                                               gamma):
            new_rs = copy.deepcopy(selected_rs)
            new_rs.append(new_rules)
            
            selected_rs = copy.deepcopy(new_rs)
    
    return selected_rs


def get_norule(rs, X, y):
    """
    Return the two smallest rule of CP1 that cover all none covered
    positive and negative points
    
    Parameters
    ----------
    rs : {RuleSet type}
         A set of rules
         
    X : {array-like or discretized matrix, shape = [n, d]}
        The training input samples after discretization.

    y : {array-like, shape = [n]}
    
        The normalized target values (real numbers).
        
    Return
    ------
    neg_rule, pos_rule : {tuple type}
                         Two rules or None
    """
    no_rule_act = 1 - rs.calc_activation()
    norule = None
    if sum(no_rule_act) > 0:
        norule_list = get_norules_list(no_rule_act, X, y)
        
        if len(norule_list) > 0:
            norule = norule_list[0]
            for rg in norule_list[1:]:
                conditions_list = norule.intersect_conditions(rg)
                new_conditions = RICE.RuleConditions(features_name=conditions_list[0],
                                                     features_index=conditions_list[1],
                                                     bmin=conditions_list[2],
                                                     bmax=conditions_list[3],
                                                     xmax=conditions_list[5],
                                                     xmin=conditions_list[4])
                norule = RICE.Rule(new_conditions)
    
    return norule


def get_norules_list(no_rule_act, X, y):
    norule_list = []
    for i in range(X.shape[1]):
        try:
            sub_x = X[:, i].astype('float')
        except ValueError:
            sub_x = None
        
        if sub_x is not None:
            sub_no_rule_act = no_rule_act[~np.isnan(sub_x)]
            sub_x = sub_x[~np.isnan(sub_x)][sub_no_rule_act]
            sub_x = np.extract(sub_no_rule_act, sub_x)
            
            norule = RICE.Rule(RICE.RuleConditions(bmin=[sub_x.min()],
                                                   bmax=[sub_x.max()],
                                                   features_name=[''],
                                                   features_index=[i],
                                                   xmax=[sub_x.max()],
                                                   xmin=[sub_x.min()]))
            norule_list.append(norule)
    
    return norule_list


def get_significant(rules_list, ymean, beta, gamma, sigma2):
    filtered_rules = filter(lambda rule: beta * abs(ymean - rule.pred) >=
                                         math.sqrt(max(0, rule.var - sigma2)), rules_list)
    significant_rules = list(filtered_rules)
    
    # print('Nb of significant rules', len(significant_rules))
    significant_rs = RICE.RuleSet(significant_rules)
    # print('Coverage rate of significant rule:', significant_rs.calc_coverage())
    
    significant_rs.sort_by(crit='cov', maximized=True)
    if len(significant_rs) > 0:
        significant_selected_rs = select_rs(rs=significant_rs, gamma=gamma)
    else:
        significant_selected_rs = RICE.RuleSet([])
    
    # print('Nb of selected rules ', len(significant_selected_rs))
    # print('Coverage rate of the selected RuleSet ', significant_selected_rs.calc_coverage())
    
    return significant_selected_rs


def add_insignificant_rules(rules_list, rs, epsilon, sigma2, gamma):
    insignificant_rule = list(filter(lambda rule: epsilon >= math.sqrt(max(0, rule.var - sigma2)),
                                     rules_list))
    # print('Nb of insignificant rules', len(insignificant_rule))
    insignificant_rs = RICE.RuleSet(insignificant_rule)
    # print('Coverage rate of significant rule:', insignificant_rs.calc_coverage())
    
    if len(insignificant_rs) > 0:
        insignificant_rs.sort_by(crit='var', maximized=False)
        selected_rs = select_rs(rs=insignificant_rs, gamma=gamma,
                                selected_rs=rs)
    else:
        selected_rs = RICE.RuleSet([])
    
    # print('Number of rules :', len(selected_rs))
    # print('Coverage rate of the selected RuleSet ', selected_rs.calc_coverage())
    
    return selected_rs


def add_norule(rs, y, X, features=None):
    new_rs = copy.deepcopy(rs)
    if rs.calc_coverage() < 1.0:
        no_rule = get_norule(copy.deepcopy(rs), X, y)
        
        if no_rule is not None:
            id_feature = no_rule.conditions.get_param('features_index')
            if features is not None:
                rule_features = list(itemgetter(*id_feature)(features))
                no_rule.conditions.set_params(features_name=rule_features)
            no_rule.calc_stats(y=y, x=X, cov_min=0.0, cov_max=1.1)
            new_rs.append(no_rule)
    
    return new_rs


def find_covering(rules_list, X, y, sigma2=None,
                  alpha=1. / 2 - 1 / 100,
                  gamma=0.95, features=None):
    
    n_train = len(y)
    cov_min = n_train ** (-alpha)
    # print('Minimal coverage rate:', cov_min)
    
    sub_rules_list = list(filter(lambda rule: rule.cov > cov_min, rules_list))
    # print('Nb of rules with good coverage rate:', len(sub_rules_list))
    
    if sigma2 is None:
        var_list = [rg.var for rg in sub_rules_list]
        sigma2 = min(list(filter(lambda v: v > 0, var_list)))
        # print('Sigma 2 estimation', sigma2)
    
    beta = pow(n_train, alpha / 2. - 1. / 4)
    # print('Beta coefficient:', beta)
    epsilon = beta * np.std(y)
    # print('Epsilon coefficient:', epsilon)
    
    significant_selected_rs = get_significant(sub_rules_list, np.mean(y), beta, gamma, sigma2)
    
    if significant_selected_rs.calc_coverage() < 1.0:
        selected_rs = add_insignificant_rules(sub_rules_list, significant_selected_rs,
                                              epsilon, sigma2, gamma)
        
        if selected_rs.calc_coverage() < 1.0:
            new_rs = add_norule(selected_rs, y, X, features)
        else:
            # print('No norule added')
            new_rs = copy.copy(selected_rs)
    else:
        # print('Significant rules form a covering')
        selected_rs = copy.copy(significant_selected_rs)
        new_rs = copy.copy(significant_selected_rs)
    
    return significant_selected_rs, selected_rs, new_rs


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
    conditions = rule.get_param('conditions').get_attr()
    length = rule.get_param('length')
    
    conditions_str = ''
    for i in range(length):
        if i > 0:
            conditions_str += ' & '
        
        conditions_str += conditions[0][i]
        if conditions[2][i] == round(conditions[3][i], 2):
            conditions_str += ' = '
            conditions_str += str(round(conditions[2][i], 2))
        else:
            conditions_str += ' $\in$ ['
            conditions_str += str(round(conditions[2][i], 2))
            conditions_str += ', '
            conditions_str += str(round(conditions[3][i], 2))
            conditions_str += ']'
    
    return conditions_str


def make_selected_df(rs):
    """
    Returns
    -------
    selected_df : {DataFrame type}
                  DataFrame of selected RuleSet for presentation
    """
    df = rs.to_df()
    
    df.rename(columns={"Cov": "Coverage", "Pred": "Prediction",
                       'Var': 'Variance', 'Crit': 'Criterium'},
              inplace=True)
    
    df['Conditions'] = [make_condition(rule) for rule in rs]
    selected_df = df[['Conditions', 'Coverage',
                      'Prediction', 'Variance',
                      'Criterium']].copy()
    
    selected_df['Coverage'] = selected_df.Coverage.round(2)
    selected_df['Prediction'] = selected_df.Prediction.round(2)
    selected_df['Variance'] = selected_df.Variance.round(2)
    selected_df['Criterium'] = selected_df.Criterium.round(2)
    
    return selected_df


def change_rs(rs, bins, xmax, xmin):
    for rg in rs:
        rg_condition = rg.conditions

        var_name = rg_condition.get_param('features_name')
        bmin = rg_condition.get_param('bmin')
        bmax = rg_condition.get_param('bmax')

        if bins is not None:
            i = 0
            for v in var_name:
                if bmin[i] > 0:
                    bmin[i] = bins[v][int(bmin[i]-1)]
                else:
                    if v == 'X0':
                        bmin[i] = xmin[0]
                    else:
                        bmin[i] = xmin[1]

                if bmax[i] < len(bins[v]):
                    bmax[i] = bins[v][int(bmax[i])]
                else:
                    if v == 'X1':
                        bmax[i] = xmax[0]
                    else:
                        bmax[i] = xmax[1]
                i += 1
                
                
def plot_rules(selected_rs, ymax, ymin,
               xmax, xmin, var1, var2,
               cm=plt.cm.RdBu, cp=None,
               bins=None):
    """
    Plot the rectangle activation zone of rules in a 2D plot
    the color is corresponding to the intensity of the prediction

    Parameters
    ----------
    var1 : {string type}
           Name of the first variable

    var2 : {string type}
           Name of the second variable

    cp : {int type}, optional
         Option to plot only the cp1 or cp2 rules

    col_pos : {string type}, optional,
              Name of the color of the zone of positive rules

    col_neg : {string type}, optional
              Name of the color of the zone of negative rules

    -------
    Draw the graphic
    """
    
    nb_color = cm.N
    selected_rs.sort_by(crit='cov', maximized=True)
    if cp is not None:
        sub_ruleset = selected_rs.extract_cp(cp)
    else:
        sub_ruleset = selected_rs
    
    plt.plot()
    
    for rg in sub_ruleset:
        rg_condition = rg.conditions
        
        var = rg_condition.get_param('features_index')
        bmin = rg_condition.get_param('bmin')
        bmax = rg_condition.get_param('bmax')
        
        cp_rg = rg.get_param('length')
        
        if rg.get_param('pred') > 0:
            hatch = '/'
            alpha = (rg.get_param('pred') / ymax)
            idx = int(nb_color / 2 + alpha * nb_color / 2) + 1
            facecolor = matplotlib.colors.rgb2hex(cm(idx))
        else:
            hatch = '\\'
            alpha = (rg.get_param('pred') / ymin)
            idx = int(nb_color / 2 - alpha * nb_color / 2) + 1
            facecolor = matplotlib.colors.rgb2hex(cm(idx))
        
        if cp_rg == 1:
            if var[0] == var1:
                p = patches.Rectangle((bmin[0], xmin[1]),  # origin
                                      abs(bmax[0] - bmin[0]),  # width
                                      xmax[1] - xmin[1],  # height
                                      hatch=hatch, facecolor=facecolor,
                                      alpha=alpha)
                plt.gca().add_patch(p)
            
            elif var[0] == var2:
                p = patches.Rectangle((xmin[0], bmin[0]),
                                      xmax[0] - xmin[0],
                                      abs(bmax[0] - bmin[0]),
                                      hatch=hatch, facecolor=facecolor,
                                      alpha=alpha)
                plt.gca().add_patch(p)
        
        elif cp_rg == 2:
            if var[0] == var1 and var[1] == var2:
                p = patches.Rectangle((bmin[0], bmin[1]),
                                      abs(bmax[0] - bmin[0]),
                                      abs(bmax[1] - bmin[1]),
                                      hatch=hatch, facecolor=facecolor,
                                      alpha=alpha)
                plt.gca().add_patch(p)
            
            elif var[1] == var1 and var[0] == var2:
                p = patches.Rectangle((bmin[1], bmin[0]),
                                      abs(bmax[1] - bmin[1]),
                                      abs(bmax[0] - bmin[0]),
                                      hatch=hatch, facecolor=facecolor,
                                      alpha=alpha)
                plt.gca().add_patch(p)
    
    if cp is None:
        plt.gca().set_title('Rules covering', fontsize=25)
    else:
        plt.gca().set_title('Rules cp%s covering' % str(cp), fontsize=25)
    
    # plt.colorbar()
    plt.gca().axis([xmin[0], xmax[0], xmin[1], xmax[1]])
