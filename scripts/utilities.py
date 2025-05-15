import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def merge_df(df1, df2, key):
    """
    Takes parameters of 2 dataframes and a key string which represents a column name both dataframes have,
    and merges the dataframes on the key string give, returning this merged dataframe.
    """
    merged_df = pd.merge(df1, df2, on=key)
    return merged_df


def one_hot_encode_df(df, x_cols):
    """
    Takes parameters df as a dataframe and x_cols as a list of categorical features, applies one hot encoding,
    and returns a dataframe with the encoded features.
    """
    onehot_encoder = OneHotEncoder(sparse_output=False)
    encoded_data = onehot_encoder.fit_transform(df[x_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out(x_cols))

    return encoded_df
