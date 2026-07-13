# CAMELYON16 Fresh-Holdout Expansion

The initial 22-slide pilot is closed and its four test slides must not be used for further tuning. This milestone adds `normal_011` through `normal_020` and `tumor_011` through `tumor_020`, producing 42 slides total.

## Predeclared split

- Train: 30 slides (15 normal, 15 tumor)
- Validation: 6 slides (`015`-`017` in each class)
- Fresh untouched test: 6 slides (`018`-`020` in each class)

The fresh six-slide test set must not be evaluated until preprocessing, model architecture, seeds, early stopping, threshold selection, and reporting rules are frozen using training and validation only.

## Current-pilot status

The completed four-slide test result remains preserved as a closed development pilot. It achieved chance-level balanced accuracy and must not be repeatedly retuned.
