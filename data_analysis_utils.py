""" 
Collection of functions for improving the predictibility of
features in a pandas dataframe including feature transformation and 
filling missing values in a pandas dataframe
"""
__author__ = "Sara Ranjbar"
__email__ = "sr110994@gmail.com"
__version__ = "0.0.1"


# to handle paths
import os

# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# for the yeo-johnson transformation
import scipy.stats as stats
 

def drop_select_vars(df, cols):
    """Drop unrelated or leaky columns."""
    for c in cols:
        if c in df.columns.values:  # ensure column exists in df
            df.drop(c, axis=1, inplace=True)
    return df


def get_target_transformation_plan(df, target):
    """
    Transforms target variable based on skewness
    using either yeojohnson or log transform.
    
    Returns df with target col transformed and the type of 
    transformation applied (to keep track of decisions for
    testset
    """
    assert target in df.columns.values
    
    res = pd.DataFrame(index=[target], columns=['transformType'])

    # see if transformation is useful based on skewness
    y = df[target]

    skewness = y.skew(axis=0)
    if abs(skewness) < 0.5:
        # target is not skewed. keep as is
        res['transformType'] = 'Skip'
    
    # ok. it is skewed. we have to transform.
    # most of the times there are 0s in target 
    # find out if that is the case
    has_zero = df[target].isin([0]).any().any()
    if has_zero:
        res['transformType'] = 'YeoJohnson'
    else:
        res['transformType'] = 'Log'

    return res
          

def categorize_vars_based_on_skewness(df, varlist):
    """
    Breaks varlist into 3 sublists: 
    notskewed, moderately skewed, extremely skewed
    
    
    Definition for these categories:
    
    -  skewness < -1 or skewness > 1 : highly skewed.
    -  -1 < skewness < -0.5 or 0.5 < skewness < 1 : moderately skewed.
    -  -0.5 < skewness < 0.5 : approximately symmetric or no skewness
    
    Returns these 3 sublists.
    """

    assert len(varlist) > 0

    skewness = df[varlist].skew(axis=0)

    moderately_skewed = []
    threshold = 0.5
    for var in varlist:
        if 0.5 <= abs(skewness[var]) <= 1.0:
            moderately_skewed.append(var)

    extremely_skewed = []
    varlist = [v for v in varlist if v not in moderately_skewed]
    threshold = 1  
    for var in varlist:
        if abs(skewness[var]) > 1.0:
            extremely_skewed.append(var)

    # catch all others
    notskewed = [v for v in varlist if var not in extremely_skewed+moderately_skewed]

    return notskewed, moderately_skewed, extremely_skewed


def get_cat_vars(df):
    return [var for var in df.columns.values if df[var].dtype == 'O']


def get_numericvars_transformation_plan(df, varlist):
    """
    Sequentially breaks varlist into bins
    based on their skewness and test a number of strategies fixes their skewness.
    
    strategies include 'Yeojohnson', and 'Log' transform. if these two dont work, we
    will move on with binarizing the remaining variables using their median.
    
    Returns transformation dataframe with index as varlist and values for 
    the column showing the decision for that column.
    """ 
    
    # ensure numeric variables are actually numeric
    # check if there are categorical variables among varlist
    # remove these from the varlist
    cat_vars = get_cat_vars(df[varlist])
    if cat_vars:
        for c in cat_vars:
            varlist.remove(c)
        
    # create result dataframe to keep track of decisions
    res = pd.DataFrame(index=varlist)
    res['transformType'] = None
    
    # test skewness in variables
    groups = categorize_vars_based_on_skewness(df, varlist)
    notskewed, modskewed, _ = groups
    print('notskewed, modskewed, extskewed:', [len(g) for g in groups])
    
    # If notskewed, no need for transformation
    if notskewed:
        res.loc[notskewed, 'transformType'] = 'Skip'
    
    # If moderately skewed, apply yeo transform
    # see if this transform fixed skewness
    print('focusing on modskewed ..')
    targetlist = modskewed
    df_tr = df[targetlist].copy()
    for var in targetlist:
        df_tr[var], _ = stats.yeojohnson(df_tr[var])
    groups = categorize_vars_based_on_skewness(df_tr, targetlist)
    fixable, nochange, worsened = groups
    print('--> fixable, nochange, worsened after yeojohnnson:', [len(g) for g in groups])
    
    # keep fixable results
    res.loc[fixable, 'transformType'] = 'YeoJohnson'
    
    # see if log transform helps no_change and worsened vars
    # make sure to only apply to cases without 0s
    print('\nfocusing on no change and worsened ones (without 0)')
    targetlist = nochange + worsened
    targetlist = [v for v in targetlist if not df[v].isin([0]).any().any()]
    targetlist = [v for v in targetlist if not (df[v] < 0).any().any()]
    df_tr = df[targetlist].copy()
    for var in targetlist:
        df_tr[var] = np.log(df_tr[var])
    groups = categorize_vars_based_on_skewness(df_tr, targetlist)
    fixable, _, _ = groups
    print('--> fixable, no_change, worsened after log:', [len(g) for g in groups])

    # update res for fixable
    res.loc[fixable, 'transformType'] = 'Log'
    
    # all remaining have to be binarized
    # you can use different criteria, we'll use medium here
    # update varlist
    print('--> if still not fixable, binarize')
    binthese = [v for v in varlist if res.loc[v, 'transformType'] is None]
    res.loc[binthese, 'transformType'] = 'Binarize'
    
    # you are done. return result
    for c in set(res['transformType'].values):
        print(c, res['transformType'].value_counts()[c])
    return res


def apply_transform_cat_var_plan(df, trans_map):
    """
    Given a transformmation plan, this function turns categorical vars 
    into numeric vars.
    """
    # ensure trans_plan is not empty
    assert type(trans_map) == dict
    
    if trans_map == {}:
        print('no plan provided. return df unchanged')
        return df
    
    for var in trans_map.keys():
        
        prevals = set(df[var].values)
        postvals = set(trans_map[var].values())
        
        # check if pre and post vals have no overlap
        if prevals.isdisjoint(postvals):
            df[var] = df[var].map(trans_map[var])
        else:
            print('%s already transformed. skip.' % var)
            continue
        
    # done. return transformed df
    return df


def apply_transform_num_var_plan(df, trans_map):
    """
    Given a transformmation plan, this function applies all the requested 
    transforms to the data.
    """
    
    # ensure trans_plan is not empty
    if trans_map.empty:
        print('no plan provided. return df unchanged')
        return df
    
    plancol = 'transformType'
    uniq_plans = list(set(trans_map[plancol].values))
    
    df_tr = df.copy()
    
    for pl in uniq_plans:
        pl_vars = trans_map[trans_map[plancol] == pl].index.values
        
        if pl == 'Skip':  # do nothing
            pass
        
        elif pl == 'YeoJohnson':
            for var in pl_vars:
                df_tr[var], _ = stats.yeojohnson(df[var])
        
        elif pl == 'Log':
            for var in pl_vars:
                df_tr[var] = np.log(df[var])
        
        elif pl == 'Binarize':
            for var in pl_vars:
                df_tr[var] = np.where(df[var]<=df[var].median(), 0, 1)
    
        else:  # do nothing
            print('transformation %s unknown. skip these' % pl)
            pass
    
    # done. return transformed df
    return df_tr

def get_fill_missing_plan(df):
    """
    Get plan for how to deal with missing values
    Returns a transformation dataframe
        - indices: variables with missing vals
        - column value for each index: what to do
    """
    # find vars with missing values
    vars_na = [v for v in df.columns.values if df[v].isnull().sum()>0]
    
    # skip if no missing vals
    if not vars_na:
        print('no missing values')
        return None
    
    # create result datafrane
    rescol = 'FillValue'
    res = pd.DataFrame(index=vars_na, columns=[rescol])
    
    # divide into cat and num variable
    cat_na = [v for v in vars_na if df[v].dtype == 'O']
    num_na = [v for v in vars_na if v not in cat_na]
    print('# cat vars with na: ', len(cat_na))
    print('# num vars with na: ', len(num_na))
    
    # for cat vars: use 0.3 as a guide/threshold (this is heuristically-decided)
    # >= 30% missing: create a new category
    # < 30% missing: replace with most commmon category
    cat_miss_th = 0.3
    perc_miss_df = df[cat_na].isnull().mean().sort_values(ascending=False)
    for v in cat_na:
        if perc_miss_df[v] >= cat_miss_th:
            res.loc[v, rescol] = 'Unknown'
        else:
            res.loc[v, rescol] = df[v].mode()
     
    ## for numeric, use 0.5 as threshold
    # >= 50% missing: drop
    # < 30% missing: replace with median
    num_miss_th = 0.5
    perc_miss_df = df[num_na].isnull().mean().sort_values(ascending=False)
    for v in num_na:
        if perc_miss_df[v] >= num_miss_th:
            res.loc[v, rescol] = 'NoFill'
        else:
            res.loc[v, rescol] = df[v].median()
    
    # return fill missing plan
    print(res.head())
    return res
    
def apply_fill_missing_plan(df, fill_plan):
    
    vars_na = fill_plan.index.values

    for v in vars_na:
        fillval = fill_plan.loc[v, 'FillValue']
        
        if fillval == 'NoFill':  # drop
            df.drop(v, axis=1, inplace=True)
        else:                  
            na_indices = list(df[df[v].isnull() == True].index.values)   
            df.loc[na_indices, v] = fillval
    
    # return new df
    return df

 