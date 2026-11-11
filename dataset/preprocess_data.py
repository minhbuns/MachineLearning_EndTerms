"""Reusable preprocessing for the Online Shoppers classification project.

No dataset is loaded and no transformer is fitted when this module is imported.
Fit each returned pipeline only on training data for predictive evaluation.
"""
from pathlib import Path
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, KBinsDiscretizer,
                                   OneHotEncoder, OrdinalEncoder, StandardScaler)

TARGET = 'Revenue'
CATEGORICAL_COLS = ['Month', 'OperatingSystems', 'Browser', 'Region',
                    'TrafficType', 'VisitorType', 'Weekend']
NUMERIC_COLS = ['Administrative', 'Administrative_Duration',
                'Informational', 'Informational_Duration',
                'ProductRelated', 'ProductRelated_Duration',
                'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']


def explore_data(data, verbose=True, remove_duplicates=True):
    """Check basic dataset properties and return a clean-index copy.

    Duplicate rows may represent separate real sessions; removing them is a
    project decision and is controlled by remove_duplicates.
    """
    data = data.copy()
    if verbose:
        print('Original shape:', data.shape)
        print('Missing values:', int(data.isna().sum().sum()))
        print('Duplicated rows:', int(data.duplicated().sum()))
        print('Target distribution (before cleaning):')
        print(data[TARGET].value_counts(dropna=False))
        print('Numeric summary:')
        print(data.describe().round(3))
    for c in NUMERIC_COLS:
        if c not in data:
            raise ValueError(f'Missing feature: {c}')
    if data.isna().any().any():
        raise ValueError('Missing values detected; impute/remove before model fitting.')
    non_negative = ['Administrative', 'Administrative_Duration',
                    'Informational', 'Informational_Duration',
                    'ProductRelated', 'ProductRelated_Duration', 'PageValues']
    ranges = ['BounceRates', 'ExitRates', 'SpecialDay']
    invalid = {c: int((data[c] < 0).sum()) for c in non_negative}
    invalid.update({c: int((~data[c].between(0, 1)).sum()) for c in ranges})
    if verbose:
        print('Out-of-range values:', invalid)
    if any(invalid.values()):
        raise ValueError('Invalid feature values detected; inspect before training.')
    if remove_duplicates:
        data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    if verbose:
        print('Cleaned shape:', data.shape)
        print('Target distribution (cleaned):')
        print(data[TARGET].value_counts())
    return data


def shift_categories(x):
    """Reserve category code 0 for unknown values, shift known codes to 1+."""
    return x + 1


def _binary():
    return ColumnTransformer([
        ('numeric', KBinsDiscretizer(n_bins=4, encode='onehot', strategy='uniform'), NUMERIC_COLS),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True), CATEGORICAL_COLS),
    ], sparse_threshold=1.0)


def _discrete():
    categorical_encoder = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ('shift', FunctionTransformer(shift_categories)),
    ])
    return ColumnTransformer([
        ('numeric', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'), NUMERIC_COLS),
        ('categorical', categorical_encoder, CATEGORICAL_COLS),
    ], sparse_threshold=0.0)


def _mixed():
    return ColumnTransformer([
        ('numeric', StandardScaler(), NUMERIC_COLS),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS),
    ], sparse_threshold=0.0)


def get_full_binary_pipeline():
    return Pipeline([('preprocessor', _binary())])


def get_full_discrete_pipeline():
    return Pipeline([('preprocessor', _discrete())])


def get_mixed_pipeline():
    return Pipeline([('preprocessor', _mixed())])


if __name__ == '__main__':
    csv_path = Path(__file__).with_name('online_shoppers_intention.csv')
    if csv_path.exists():
        explore_data(pd.read_csv(csv_path))
    else:
        print('Place online_shoppers_intention.csv next to this file to preview EDA.')
