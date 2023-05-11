
""" 
Unittesting data_analysis_utils functions
"""

__author__ = "Sara Ranjbar"
__email__ = "sr110994@gmail.com"
__version__ = "0.0.1"

from data_analysis_utils import *

def test_fill_missing(df):
    
    tmp = fill_in_missing_vals(df)
    vars_na = [v for v in tmp.columns.values if tmp[v].isnull().sum()>0]
    assert len(vars_na) == 0


def test_drop_select_vars(df, drop_vars):
    
    tmp = drop_select_vars(df, drop_vars)
    for c in drop_vars:
        assert c not in tmp.columns.values

def test_transform_vat_vars(df, mapping):

    tmp = transform_cat_vars(df, mapping)
    for var in mapping.keys():
        
        prevals  = set(df[var].values)
        postvals = set(tmp[var].values)
        assert prevals != postvals
        assert prevals not in postvals