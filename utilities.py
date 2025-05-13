import pandas as pd;

def merge_df(df1, df2, key):
    """
    takes parameters of 2 dataframes and a key string which represents a column name both dataframes have,
    and merges the dataframes on the key string give, returning this merged dataframe.
    """
    merged_df = pd.merge(df1, df2, on=key)
    return merged_df