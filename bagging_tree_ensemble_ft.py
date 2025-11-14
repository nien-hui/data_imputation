
from typing import Tuple, Iterable, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif


# def _ensure_numeric_df(train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str):
#     if label_col not in train_df.columns:
#         raise ValueError(f"train_df must contain the label column '{label_col}'.")

#     y_train = train_df[label_col].copy()
#     y_test = test_df[label_col].copy() if label_col in test_df.columns else None

#     X_train = train_df.drop(columns=[label_col])
#     X_test = test_df.drop(columns=[label_col]) if y_test is not None else test_df.copy()

#     X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
#     non_numeric_cols = X_all.select_dtypes(exclude=[np.number]).columns.tolist()
#     if non_numeric_cols:
#         X_all = pd.get_dummies(X_all, columns=non_numeric_cols, dummy_na=False)

#     X_train_num = X_all.iloc[: len(X_train), :].reset_index(drop=True)
#     X_test_num = X_all.iloc[len(X_train) :, :].reset_index(drop=True)

#     return X_train_num, X_test_num, y_train.reset_index(drop=True), (y_test.reset_index(drop=True) if y_test is not None else None)

def _ensure_numeric_df(train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str, nan_value):
    if label_col not in train_df.columns:
        raise ValueError(f"train_df must contain the label column '{label_col}'.")

    y_train = train_df[label_col].copy()
    y_test = test_df[label_col].copy() if label_col in test_df.columns else None

    X_train = train_df.drop(columns=[label_col])
    X_test = test_df.drop(columns=[label_col]) if y_test is not None else test_df.copy()

    X_train = X_train.fillna(nan_value)
    X_test = X_test.fillna(nan_value)

    X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # 仍然對非數值欄位做 one-hot
    non_numeric_cols = X_all.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        X_all = pd.get_dummies(X_all, columns=non_numeric_cols, dummy_na=False)

    X_train_num = X_all.iloc[: len(X_train), :].reset_index(drop=True)
    X_test_num = X_all.iloc[len(X_train) :, :].reset_index(drop=True)

    return X_train_num, X_test_num, y_train.reset_index(drop=True), (y_test.reset_index(drop=True) if y_test is not None else None)

def _fit_model(model_key: str,
               n_estimators: int,
               random_state: int = 42,
               max_depth: Optional[int] = None,
               max_leaf_nodes: Optional[int] = 64,
               min_samples_leaf: int = 1):
    if model_key == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            criterion="gini",
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
        )
    elif model_key == "bagging_dt":
        base = DecisionTreeClassifier(
            criterion="gini",
            random_state=random_state,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
        )
        return BaggingClassifier(
            estimator=base,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model key '{model_key}'. Use 'rf' or 'bagging_dt'.")


def _collect_leaf_matrix(est, X: pd.DataFrame):
    """Return an (n_samples, n_trees) matrix of leaf ids for the ensemble."""
    if hasattr(est, "apply"):
        leaves = est.apply(X)
        if isinstance(leaves, list):
            leaves = np.asarray(leaves)
        if leaves.ndim == 1:
            if hasattr(est, "estimators_"):
                leaves = np.column_stack([tree.apply(X) for tree in est.estimators_])
            else:
                leaves = leaves.reshape(-1, 1)
        return leaves
    if hasattr(est, "estimators_"):
        return np.column_stack([tree.apply(X) for tree in est.estimators_])
    raise AttributeError(f"Estimator of type {type(est).__name__} does not expose leaf indices.")


def _leaf_one_hot_block(est, X_train: pd.DataFrame, X_test: pd.DataFrame, prefix: str):
    train_leaves = _collect_leaf_matrix(est, X_train)
    test_leaves = _collect_leaf_matrix(est, X_test)

    if train_leaves.ndim == 1:
        tr_cols, te_cols = [], []
        for tree in est.estimators_:
            tr_cols.append(tree.apply(X_train).reshape(-1, 1))
            te_cols.append(tree.apply(X_test).reshape(-1, 1))
        train_leaves = np.hstack(tr_cols)
        test_leaves = np.hstack(te_cols)

    n_trees = train_leaves.shape[1]

    tr_blocks = []
    te_blocks = []

    for t in range(n_trees):
        # Use sparse_output=True for modern sklearn. No 'dtype' arg to keep compatibility.
        enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        tr_vec = train_leaves[:, [t]]
        te_vec = test_leaves[:, [t]]
        tr_onehot = enc.fit_transform(tr_vec)
        te_onehot = enc.transform(te_vec)

        cats = enc.categories_[0].astype(int)
        tr_blocks.append((tr_onehot, [f"{prefix}_t{t}_leaf{c}" for c in cats]))
        te_blocks.append((te_onehot, [f"{prefix}_t{t}_leaf{c}" for c in cats]))

    tr_sparse = sp.hstack([b for (b, _) in tr_blocks], format="csr")
    te_sparse = sp.hstack([b for (b, _) in te_blocks], format="csr")
    col_names = sum([names for (_, names) in tr_blocks], [])

    return tr_sparse, te_sparse, col_names


def bagging_tree_ensemble_ft(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "label",
    models: Iterable[str] = ("rf", "bagging_dt"),
    n_estimators: int = 100,
    top_ratio: float = 0.10,
    random_state: int = 42,
    max_depth: Optional[int] = None,
    max_leaf_nodes: Optional[int] = 64,
    min_samples_leaf: int = 1,
    cast_float32_for_mi: bool = True,
    id_col: Optional[str] = "SK_ID_CURR",
    nan_value = -99999
):
    train_ids = train_df[id_col].reset_index(drop=True) if id_col and id_col in train_df.columns else None
    test_ids = test_df[id_col].reset_index(drop=True) if id_col and id_col in test_df.columns else None

    if id_col and id_col in train_df.columns:
        train_df = train_df.drop(columns=[id_col])
    if id_col and id_col in test_df.columns:
        test_df = test_df.drop(columns=[id_col])

    X_train, X_test, y_train, y_test_optional = _ensure_numeric_df(train_df, test_df, label_col, nan_value)

    tr_blocks = []
    te_blocks = []
    name_blocks = []

    for key in models:
        model = _fit_model(key,
                           n_estimators=n_estimators,
                           random_state=random_state,
                           max_depth=max_depth,
                           max_leaf_nodes=max_leaf_nodes,
                           min_samples_leaf=min_samples_leaf)
        model.fit(X_train, y_train)
        tr_sp, te_sp, cols = _leaf_one_hot_block(model, X_train, X_test, prefix=key)
        tr_blocks.append(tr_sp)
        te_blocks.append(te_sp)
        name_blocks.extend(cols)

    X_train_leaf = sp.hstack(tr_blocks, format="csr")
    X_test_leaf = sp.hstack(te_blocks, format="csr")

    X_mi = X_train_leaf.astype(np.float32) if cast_float32_for_mi else X_train_leaf
    mi = mutual_info_classif(X_mi, y_train.values, discrete_features=True, random_state=random_state)

    k = max(1, int(np.ceil(len(mi) * float(top_ratio))))
    top_idx = np.argsort(mi)[-k:]
    selected_names = [name_blocks[i] for i in top_idx]

    X_train_sel = X_train_leaf[:, top_idx]
    X_test_sel = X_test_leaf[:, top_idx]

    train_dense = pd.DataFrame(X_train_sel.toarray(), columns=selected_names, index=X_train.index)
    test_dense = pd.DataFrame(X_test_sel.toarray(), columns=selected_names, index=X_test.index)

    train_df_bagging = train_dense.reset_index(drop=True)
    if train_ids is not None:
        train_df_bagging.insert(0, id_col, train_ids)
    train_df_bagging = pd.concat([train_df_bagging, y_train.reset_index(drop=True)], axis=1)

    if y_test_optional is not None:
        test_df_bagging = test_dense.reset_index(drop=True)
        if test_ids is not None:
            test_df_bagging.insert(0, id_col, test_ids)
        test_df_bagging = pd.concat([test_df_bagging, y_test_optional.reset_index(drop=True)], axis=1)
    else:
        test_df_bagging = test_dense.reset_index(drop=True)
        if test_ids is not None:
            test_df_bagging.insert(0, id_col, test_ids)

    return train_df_bagging, test_df_bagging
