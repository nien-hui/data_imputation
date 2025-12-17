# -*- coding: utf-8 -*-
import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# Configs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MY_DIR = os.path.join(BASE_DIR, "my_data")
default_dir = MY_DIR

# -----------------------------------------------------------------------------
# Data Loading & Helper Functions
# -----------------------------------------------------------------------------

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Mem usage optimized: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / max(start_mem, 1e-9):.1f}% reduction)")
    return df

def get_balance_data():
    # Define dtypes to save memory on load
    pos_dtype = {
        "SK_ID_PREV": np.uint32, "SK_ID_CURR": np.uint32, "MONTHS_BALANCE": np.int32,
        "SK_DPD": np.int32, "SK_DPD_DEF": np.int32, "CNT_INSTALMENT": np.float32,
        "CNT_INSTALMENT_FUTURE": np.float32,
    }
    install_dtype = {
        "SK_ID_PREV": np.uint32, "SK_ID_CURR": np.uint32, "NUM_INSTALMENT_NUMBER": np.int32,
        "NUM_INSTALMENT_VERSION": np.float32, "DAYS_INSTALMENT": np.float32,
        "DAYS_ENTRY_PAYMENT": np.float32, "AMT_INSTALMENT": np.float32, "AMT_PAYMENT": np.float32,
    }
    card_dtype = {
        "SK_ID_PREV": np.uint32, "SK_ID_CURR": np.uint32, "MONTHS_BALANCE": np.int16,
        "AMT_CREDIT_LIMIT_ACTUAL": np.int32, "CNT_DRAWINGS_CURRENT": np.int32,
        "SK_DPD": np.int32, "SK_DPD_DEF": np.int32, "AMT_BALANCE": np.float32,
        "AMT_DRAWINGS_ATM_CURRENT": np.float32, "AMT_DRAWINGS_CURRENT": np.float32,
        "AMT_DRAWINGS_OTHER_CURRENT": np.float32, "AMT_DRAWINGS_POS_CURRENT": np.float32,
        "AMT_INST_MIN_REGULARITY": np.float32, "AMT_PAYMENT_CURRENT": np.float32,
        "AMT_PAYMENT_TOTAL_CURRENT": np.float32, "AMT_RECEIVABLE_PRINCIPAL": np.float32,
        "AMT_RECIVABLE": np.float32, "AMT_TOTAL_RECEIVABLE": np.float32,
        "CNT_DRAWINGS_ATM_CURRENT": np.float32, "CNT_DRAWINGS_OTHER_CURRENT": np.float32,
        "CNT_DRAWINGS_POS_CURRENT": np.float32, "CNT_INSTALMENT_MATURE_CUM": np.float32,
    }

    pos_bal = pd.read_csv(os.path.join(DATA_DIR, "POS_CASH_balance.csv"), dtype=pos_dtype)
    install = pd.read_csv(os.path.join(DATA_DIR, "installments_payments.csv"), dtype=install_dtype)
    card_bal = pd.read_csv(os.path.join(DATA_DIR, "credit_card_balance.csv"), dtype=card_dtype)
    return pos_bal, install, card_bal

# -----------------------------------------------------------------------------
# Feature Engineering & Aggregation
# -----------------------------------------------------------------------------

def get_apps_processed(apps: pd.DataFrame) -> pd.DataFrame:
    # External sources
    apps["APPS_EXT_SOURCE_MEAN"] = apps[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    apps["APPS_EXT_SOURCE_STD"] = apps[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    apps["APPS_EXT_SOURCE_STD"] = apps["APPS_EXT_SOURCE_STD"].fillna(apps["APPS_EXT_SOURCE_STD"].mean())

    # Credit/Income ratios
    apps["APPS_ANNUITY_CREDIT_RATIO"] = apps["AMT_ANNUITY"] / apps["AMT_CREDIT"]
    apps["APPS_GOODS_CREDIT_RATIO"] = apps["AMT_GOODS_PRICE"] / apps["AMT_CREDIT"]
    apps["APPS_ANNUITY_INCOME_RATIO"] = apps["AMT_ANNUITY"] / apps["AMT_INCOME_TOTAL"]
    apps["APPS_CREDIT_INCOME_RATIO"] = apps["AMT_CREDIT"] / apps["AMT_INCOME_TOTAL"]
    apps["APPS_GOODS_INCOME_RATIO"] = apps["AMT_GOODS_PRICE"] / apps["AMT_INCOME_TOTAL"]
    apps["APPS_CNT_FAM_INCOME_RATIO"] = apps["AMT_INCOME_TOTAL"] / apps["CNT_FAM_MEMBERS"]

    # Age/Employment ratios
    apps["APPS_EMPLOYED_BIRTH_RATIO"] = apps["DAYS_EMPLOYED"] / apps["DAYS_BIRTH"]
    apps["APPS_INCOME_EMPLOYED_RATIO"] = apps["AMT_INCOME_TOTAL"] / apps["DAYS_EMPLOYED"]
    apps["APPS_INCOME_BIRTH_RATIO"] = apps["AMT_INCOME_TOTAL"] / apps["DAYS_BIRTH"]
    apps["APPS_CAR_BIRTH_RATIO"] = apps["OWN_CAR_AGE"] / apps["DAYS_BIRTH"]
    apps["APPS_CAR_EMPLOYED_RATIO"] = apps["OWN_CAR_AGE"] / apps["DAYS_EMPLOYED"]
    return apps

def get_prev_processed(prev: pd.DataFrame) -> pd.DataFrame:
    prev["PREV_CREDIT_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    prev["PREV_GOODS_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_GOODS_PRICE"]
    prev["PREV_CREDIT_APPL_RATIO"] = prev["AMT_CREDIT"] / prev["AMT_APPLICATION"]
    prev["PREV_GOODS_APPL_RATIO"] = prev["AMT_GOODS_PRICE"] / prev["AMT_APPLICATION"]

    # Cleaning outliers
    for col in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        prev[col].replace(365243, np.nan, inplace=True)

    prev["PREV_DAYS_LAST_DUE_DIFF"] = prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_LAST_DUE"]
    
    # Interest rate approx
    all_pay = prev["AMT_ANNUITY"] * prev["CNT_PAYMENT"]
    prev["PREV_INTERESTS_RATE"] = (all_pay / prev["AMT_CREDIT"] - 1) / prev["CNT_PAYMENT"]
    return prev

def get_prev_agg(prev: pd.DataFrame) -> pd.DataFrame:
    prev = get_prev_processed(prev)
    
    # General Amount Aggregation
    agg_dict = {
        "SK_ID_CURR": ["count"],
        "AMT_CREDIT": ["mean", "max", "sum"],
        "AMT_ANNUITY": ["mean", "max", "sum"],
        "AMT_APPLICATION": ["mean", "max", "sum"],
        "AMT_DOWN_PAYMENT": ["mean", "max", "sum"],
        "AMT_GOODS_PRICE": ["mean", "max", "sum"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
        "PREV_CREDIT_DIFF": ["mean", "max", "sum"],
        "PREV_CREDIT_APPL_RATIO": ["mean", "max"],
        "PREV_GOODS_DIFF": ["mean", "max", "sum"],
        "PREV_GOODS_APPL_RATIO": ["mean", "max"],
        "PREV_DAYS_LAST_DUE_DIFF": ["mean", "max", "sum"],
        "PREV_INTERESTS_RATE": ["mean", "max"],
    }
    prev_amt_agg = prev.groupby("SK_ID_CURR").agg(agg_dict)
    prev_amt_agg.columns = ["PREV_" + "_".join(x).upper() for x in prev_amt_agg.columns.ravel()]

    # Approved/Refused counts
    mask = prev["NAME_CONTRACT_STATUS"].isin(["Approved", "Refused"])
    prev_refused_appr_agg = prev[mask].groupby(["SK_ID_CURR", "NAME_CONTRACT_STATUS"])["SK_ID_CURR"].count().unstack().fillna(0)
    prev_refused_appr_agg.columns = ["PREV_APPROVED_COUNT", "PREV_REFUSED_COUNT"]

    # Last 365 days Aggregation
    prev_days365_agg = prev[prev["DAYS_DECISION"] > -365].groupby("SK_ID_CURR").agg(agg_dict)
    prev_days365_agg.columns = ["PREV_D365_" + "_".join(x).upper() for x in prev_days365_agg.columns.ravel()]

    # Merge
    prev_agg = prev_amt_agg.merge(prev_refused_appr_agg, on="SK_ID_CURR", how="left")
    prev_agg = prev_agg.merge(prev_days365_agg, on="SK_ID_CURR", how="left")
    
    # Ratios
    prev_agg["PREV_REFUSED_RATIO"] = prev_agg["PREV_REFUSED_COUNT"] / prev_agg["PREV_SK_ID_CURR_COUNT"]
    prev_agg["PREV_APPROVED_RATIO"] = prev_agg["PREV_APPROVED_COUNT"] / prev_agg["PREV_SK_ID_CURR_COUNT"]
    prev_agg = prev_agg.drop(["PREV_REFUSED_COUNT", "PREV_APPROVED_COUNT"], axis=1)

    return prev_agg

def get_bureau_processed(bureau: pd.DataFrame) -> pd.DataFrame:
    bureau["BUREAU_ENDDATE_FACT_DIFF"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_ENDDATE_FACT"]
    bureau["BUREAU_CREDIT_FACT_DIFF"] = bureau["DAYS_CREDIT"] - bureau["DAYS_ENDDATE_FACT"]
    bureau["BUREAU_CREDIT_ENDDATE_DIFF"] = bureau["DAYS_CREDIT"] - bureau["DAYS_CREDIT_ENDDATE"]
    bureau["BUREAU_CREDIT_DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM"]
    bureau["BUREAU_CREDIT_DEBT_DIFF"] = bureau["AMT_CREDIT_SUM_DEBT"] - bureau["AMT_CREDIT_SUM"]
    bureau["BUREAU_IS_DPD"] = (bureau["CREDIT_DAY_OVERDUE"] > 0).astype(int)
    bureau["BUREAU_IS_DPD_OVER120"] = (bureau["CREDIT_DAY_OVERDUE"] > 120).astype(int)
    return bureau

def get_bureau_agg(bureau: pd.DataFrame, bureau_bal: pd.DataFrame) -> pd.DataFrame:
    bureau = get_bureau_processed(bureau)
    
    agg_dict = {
        "SK_ID_BUREAU": ["count"],
        "DAYS_CREDIT": ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_ENDDATE_FACT": ["min", "max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["max", "mean", "sum"],
        "AMT_ANNUITY": ["max", "mean", "sum"],
        "BUREAU_ENDDATE_FACT_DIFF": ["min", "max", "mean"],
        "BUREAU_CREDIT_FACT_DIFF": ["min", "max", "mean"],
        "BUREAU_CREDIT_ENDDATE_DIFF": ["min", "max", "mean"],
        "BUREAU_CREDIT_DEBT_RATIO": ["min", "max", "mean"],
        "BUREAU_CREDIT_DEBT_DIFF": ["min", "max", "mean"],
        "BUREAU_IS_DPD": ["mean", "sum"],
        "BUREAU_IS_DPD_OVER120": ["mean", "sum"],
    }

    # 1. All records agg
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(agg_dict)
    bureau_agg.columns = ["BUREAU_" + "_".join(x).upper() for x in bureau_agg.columns.ravel()]
    bureau_agg.reset_index(inplace=True)

    # 2. Active credits agg
    active_agg = bureau[bureau["CREDIT_ACTIVE"] == "Active"].groupby("SK_ID_CURR").agg(agg_dict)
    active_agg.columns = ["BUREAU_ACT_" + "_".join(x).upper() for x in active_agg.columns.ravel()]
    active_agg.reset_index(inplace=True)

    # 3. Recent (750 days) agg
    days750_agg = bureau[bureau["DAYS_CREDIT"] > -750].groupby("SK_ID_CURR").agg(agg_dict)
    days750_agg.columns = ["BUREAU_ACT_" + "_".join(x).upper() for x in days750_agg.columns.ravel()]
    days750_agg.reset_index(inplace=True)

    # 4. Bureau Balance Agg
    bureau_bal = bureau_bal.merge(bureau[["SK_ID_CURR", "SK_ID_BUREAU"]], on="SK_ID_BUREAU", how="left")
    bureau_bal["BUREAU_BAL_IS_DPD"] = bureau_bal["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
    bureau_bal["BUREAU_BAL_IS_DPD_OVER120"] = (bureau_bal["STATUS"] == "5").astype(int)
    
    bal_agg = bureau_bal.groupby("SK_ID_CURR").agg({
        "SK_ID_CURR": ["count"], "MONTHS_BALANCE": ["min", "max", "mean"],
        "BUREAU_BAL_IS_DPD": ["mean", "sum"], "BUREAU_BAL_IS_DPD_OVER120": ["mean", "sum"]
    })
    bal_agg.columns = ["BUREAU_BAL_" + "_".join(x).upper() for x in bal_agg.columns.ravel()]
    bal_agg.reset_index(inplace=True)

    # Final Merge
    bureau_agg = bureau_agg.merge(active_agg, on="SK_ID_CURR", how="left")
    bureau_agg["BUREAU_ACT_IS_DPD_RATIO"] = bureau_agg["BUREAU_ACT_BUREAU_IS_DPD_SUM"] / bureau_agg["BUREAU_SK_ID_BUREAU_COUNT"]
    bureau_agg["BUREAU_ACT_IS_DPD_OVER120_RATIO"] = bureau_agg["BUREAU_ACT_BUREAU_IS_DPD_OVER120_SUM"] / bureau_agg["BUREAU_SK_ID_BUREAU_COUNT"]
    
    bureau_agg = bureau_agg.merge(bal_agg, on="SK_ID_CURR", how="left")
    bureau_agg = bureau_agg.merge(days750_agg, on="SK_ID_CURR", how="left")
    return bureau_agg

def get_pos_bal_agg(pos_bal: pd.DataFrame) -> pd.DataFrame:
    pos_bal["POS_IS_DPD"] = (pos_bal["SK_DPD"] > 0).astype(int)
    pos_bal["POS_IS_DPD_UNDER_120"] = ((pos_bal["SK_DPD"] > 0) & (pos_bal["SK_DPD"] < 120)).astype(int)
    pos_bal["POS_IS_DPD_OVER_120"] = (pos_bal["SK_DPD"] >= 120).astype(int)

    agg_dict = {
        "SK_ID_CURR": ["count"], "MONTHS_BALANCE": ["min", "mean", "max"],
        "SK_DPD": ["min", "max", "mean", "sum"], "CNT_INSTALMENT": ["min", "max", "mean", "sum"],
        "CNT_INSTALMENT_FUTURE": ["min", "max", "mean", "sum"],
        "POS_IS_DPD": ["mean", "sum"], "POS_IS_DPD_UNDER_120": ["mean", "sum"], "POS_IS_DPD_OVER_120": ["mean", "sum"],
    }
    
    pos_agg = pos_bal.groupby("SK_ID_CURR").agg(agg_dict)
    pos_agg.columns = ["POS_" + "_".join(x).upper() for x in pos_agg.columns.ravel()]
    
    pos_m20_agg = pos_bal[pos_bal["MONTHS_BALANCE"] > -20].groupby("SK_ID_CURR").agg(agg_dict)
    pos_m20_agg.columns = ["POS_M20" + "_".join(x).upper() for x in pos_m20_agg.columns.ravel()]

    pos_agg = pos_agg.merge(pos_m20_agg, on="SK_ID_CURR", how="left").reset_index()
    return pos_agg

def get_install_agg(install: pd.DataFrame) -> pd.DataFrame:
    install["AMT_DIFF"] = install["AMT_INSTALMENT"] - install["AMT_PAYMENT"]
    install["AMT_RATIO"] = (install["AMT_PAYMENT"] + 1) / (install["AMT_INSTALMENT"] + 1)
    install["SK_DPD"] = install["DAYS_ENTRY_PAYMENT"] - install["DAYS_INSTALMENT"]
    install["INS_IS_DPD"] = (install["SK_DPD"] > 0).astype(int)
    install["INS_IS_DPD_UNDER_120"] = ((install["SK_DPD"] > 0) & (install["SK_DPD"] < 120)).astype(int)
    install["INS_IS_DPD_OVER_120"] = (install["SK_DPD"] >= 120).astype(int)

    agg_dict = {
        "SK_ID_CURR": ["count"], "NUM_INSTALMENT_VERSION": ["nunique"],
        "DAYS_ENTRY_PAYMENT": ["mean", "max", "sum"], "DAYS_INSTALMENT": ["mean", "max", "sum"],
        "AMT_INSTALMENT": ["mean", "max", "sum"], "AMT_PAYMENT": ["mean", "max", "sum"],
        "AMT_DIFF": ["mean", "min", "max", "sum"], "AMT_RATIO": ["mean", "max"],
        "SK_DPD": ["mean", "min", "max"], "INS_IS_DPD": ["mean", "sum"],
        "INS_IS_DPD_UNDER_120": ["mean", "sum"], "INS_IS_DPD_OVER_120": ["mean", "sum"],
    }
    
    ins_agg = install.groupby("SK_ID_CURR").agg(agg_dict)
    ins_agg.columns = ["INS_" + "_".join(x).upper() for x in ins_agg.columns.ravel()]

    ins_d365_agg = install[install["DAYS_ENTRY_PAYMENT"] >= -365].groupby("SK_ID_CURR").agg(agg_dict)
    ins_d365_agg.columns = ["INS_D365" + "_".join(x).upper() for x in ins_d365_agg.columns.ravel()]

    ins_agg = ins_agg.merge(ins_d365_agg, on="SK_ID_CURR", how="left").reset_index()
    return ins_agg

def get_card_bal_agg(card_bal: pd.DataFrame) -> pd.DataFrame:
    card_bal["BALANCE_LIMIT_RATIO"] = card_bal["AMT_BALANCE"] / card_bal["AMT_CREDIT_LIMIT_ACTUAL"]
    card_bal["DRAWING_LIMIT_RATIO"] = card_bal["AMT_DRAWINGS_CURRENT"] / card_bal["AMT_CREDIT_LIMIT_ACTUAL"]
    card_bal["CARD_IS_DPD"] = (card_bal["SK_DPD"] > 0).astype(int)
    card_bal["CARD_IS_DPD_UNDER_120"] = ((card_bal["SK_DPD"] > 0) & (card_bal["SK_DPD"] < 120)).astype(int)
    card_bal["CARD_IS_DPD_OVER_120"] = (card_bal["SK_DPD"] >= 120).astype(int)

    agg_dict = {
        "SK_ID_CURR": ["count"], "AMT_BALANCE": ["max"], "AMT_CREDIT_LIMIT_ACTUAL": ["max"],
        "AMT_DRAWINGS_ATM_CURRENT": ["max", "sum"], "AMT_DRAWINGS_CURRENT": ["max", "sum"],
        "AMT_DRAWINGS_POS_CURRENT": ["max", "sum"], "AMT_INST_MIN_REGULARITY": ["max", "mean"],
        "AMT_PAYMENT_TOTAL_CURRENT": ["max", "sum"], "AMT_TOTAL_RECEIVABLE": ["max", "mean"],
        "CNT_DRAWINGS_ATM_CURRENT": ["max", "sum"], "CNT_DRAWINGS_CURRENT": ["max", "mean", "sum"],
        "CNT_DRAWINGS_POS_CURRENT": ["mean"], "SK_DPD": ["mean", "max", "sum"],
        "BALANCE_LIMIT_RATIO": ["min", "max"], "DRAWING_LIMIT_RATIO": ["min", "max"],
        "CARD_IS_DPD": ["mean", "sum"], "CARD_IS_DPD_UNDER_120": ["mean", "sum"],
        "CARD_IS_DPD_OVER_120": ["mean", "sum"],
    }

    card_agg = card_bal.groupby("SK_ID_CURR").agg(agg_dict)
    card_agg.columns = ["CARD_" + "_".join(x).upper() for x in card_agg.columns.ravel()]
    card_agg.reset_index(inplace=True)

    card_m3_agg = card_bal[card_bal.MONTHS_BALANCE >= -3].groupby("SK_ID_CURR").agg(agg_dict)
    card_m3_agg.columns = ["CARD_M3" + "_".join(x).upper() for x in card_m3_agg.columns.ravel()]

    card_agg = card_agg.merge(card_m3_agg, on="SK_ID_CURR", how="left").reset_index()
    return card_agg

# -----------------------------------------------------------------------------
# Main Feature Integration
# -----------------------------------------------------------------------------

def build_my_features():
    print("Building custom features (Imputed + Bagging + Clusters)...")
    
    # Load raw custom data
    app_train = pd.read_csv(os.path.join(MY_DIR, "application_train.csv"))
    app_test = pd.read_csv(os.path.join(MY_DIR, "application_test.csv"))
    imputed_train = pd.read_csv(os.path.join(MY_DIR, "inputed_train_test", "imputed_train.csv"))
    imputed_test = pd.read_csv(os.path.join(MY_DIR, "inputed_train_test", "imputed_test.csv"))
    bagging_train = pd.read_csv(os.path.join(MY_DIR, "tree_ensemble_with_imputed_train_test", "train_df_bagging.csv"))
    bagging_test = pd.read_csv(os.path.join(MY_DIR, "tree_ensemble_with_imputed_train_test", "test_df_bagging.csv"))
    cluster_all = pd.read_csv(os.path.join(MY_DIR, "feature_cluster_labels_new.csv"))

    # Cleanup
    for df in [app_train, app_test, imputed_train, imputed_test, bagging_train, bagging_test, cluster_all]:
        df.columns = df.columns.str.strip()
    
    bagging_train.drop(columns=["TARGET"], inplace=True, errors='ignore')
    bagging_test.drop(columns=["TARGET"], inplace=True, errors='ignore')

    # Split clusters
    cluster_train = cluster_all[cluster_all["SK_ID_CURR"].isin(app_train["SK_ID_CURR"])]
    cluster_test = cluster_all[cluster_all["SK_ID_CURR"].isin(app_test["SK_ID_CURR"])]

    # Merge Base
    base_train = imputed_train.merge(bagging_train, on="SK_ID_CURR", how="left").merge(cluster_train, on="SK_ID_CURR", how="left")
    base_test = imputed_test.merge(bagging_test, on="SK_ID_CURR", how="left").merge(cluster_test, on="SK_ID_CURR", how="left")

    # Extract only new features
    orig_train_cols = set(app_train.columns)
    extra_cols = [c for c in base_train.columns if (c not in orig_train_cols) and (c != "TARGET")]
    
    print(f"New custom features count: {len(extra_cols)}")

    # Ensure consistency
    missing_in_test = set(extra_cols) - set(base_test.columns)
    for col in missing_in_test:
        base_test[col] = pd.NA

    my_features_train = base_train[["SK_ID_CURR"] + extra_cols]
    my_features_test = base_test[["SK_ID_CURR"] + extra_cols]

    my_features_train.to_csv(os.path.join(MY_DIR, "my_features_train.csv"), index=False)
    my_features_test.to_csv(os.path.join(MY_DIR, "my_features_test.csv"), index=False)
    print("Custom features saved.")

def get_dataset():
    print("Loading full dataset...")
    # Load Apps (Original + Custom)
    app_train = reduce_mem_usage(pd.read_csv(os.path.join(DATA_DIR, "application_train.csv")))
    app_test = reduce_mem_usage(pd.read_csv(os.path.join(DATA_DIR, "application_test.csv")))
    
    my_train = pd.read_csv(os.path.join(MY_DIR, "my_features_train.csv"))
    my_test = pd.read_csv(os.path.join(MY_DIR, "my_features_test.csv"))
    
    app_train = app_train.merge(my_train, on="SK_ID_CURR", how="left")
    app_test = app_test.merge(my_test, on="SK_ID_CURR", how="left")
    apps = pd.concat([app_train, app_test], ignore_index=True)

    # Load Tables
    prev = reduce_mem_usage(pd.read_csv(os.path.join(DATA_DIR, "previous_application.csv")))
    bureau = reduce_mem_usage(pd.read_csv(os.path.join(DATA_DIR, "bureau.csv")))
    bureau_bal = reduce_mem_usage(pd.read_csv(os.path.join(DATA_DIR, "bureau_balance.csv")))
    pos_bal, install, card_bal = get_balance_data()
    
    return apps, prev, bureau, bureau_bal, pos_bal, install, card_bal

def get_apps_all_with_all_agg(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal):
    apps_all = get_apps_processed(apps)
    
    # Merge aggregations
    print("Merging aggregations...")
    apps_all = apps_all.merge(get_prev_agg(prev), on="SK_ID_CURR", how="left")
    apps_all = apps_all.merge(get_bureau_agg(bureau, bureau_bal), on="SK_ID_CURR", how="left")
    apps_all = apps_all.merge(get_pos_bal_agg(pos_bal), on="SK_ID_CURR", how="left")
    apps_all = apps_all.merge(get_install_agg(install), on="SK_ID_CURR", how="left")
    apps_all = apps_all.merge(get_card_bal_agg(card_bal), on="SK_ID_CURR", how="left")
    
    print(f"Final dataset shape: {apps_all.shape}")
    return apps_all

def get_apps_all_encoded(apps_all: pd.DataFrame) -> pd.DataFrame:
    for col in apps_all.select_dtypes(include=['object']).columns:
        apps_all[col] = pd.factorize(apps_all[col])[0]
    return apps_all

# -----------------------------------------------------------------------------
# Training & Tuning
# -----------------------------------------------------------------------------

def tune_lgbm_random_search(train_x, train_y, valid_x, valid_y, n_iter=30, random_state=2020, desc=""):
    print(f"LGBM Random Search: {desc}")
    
    param_dist = {
        "num_leaves": [32, 48, 64, 80, 96],
        "max_depth": [7, 9, 11, -1],
        "min_child_samples": [60, 100, 140, 180],
        "min_child_weight": [1.0, 5.0, 10.0],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "reg_alpha": [0.0, 0.5, 1.0, 2.0, 3.0],
        "reg_lambda": [0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
        "max_bin": [255, 355, 455],
        "learning_rate": [0.015, 0.02, 0.03],
        "n_estimators": [2000, 3000, 4000],
    }

    sampler = ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state)
    best_auc, best_params, best_model = -1.0, None, None

    for i, params in enumerate(sampler, 1):
        clf = LGBMClassifier(objective="binary", nthread=4, **params)
        clf.fit(
            train_x, train_y, 
            eval_set=[(valid_x, valid_y)], eval_metric="auc",
            callbacks=[early_stopping(stopping_rounds=200), log_evaluation(period=0)] # Quiet log
        )
        
        valid_pred = clf.predict_proba(valid_x)[:, 1]
        auc = roc_auc_score(valid_y, valid_pred)
        
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = clf
            print(f"[Iter {i}] New Best AUC: {best_auc:.6f}")

    return best_model, best_params, best_auc

def tune_lgbm_separate_base_and_ext(apps_all_train, extra_feature_cols, n_iter=12):
    y = apps_all_train["TARGET"].values
    X_full = apps_all_train.drop(["SK_ID_CURR", "TARGET"], axis=1)
    
    base_cols = [c for c in X_full.columns if c not in set(extra_feature_cols)]
    X_base = X_full[base_cols]
    X_ext = X_full

    Xb_tr, Xb_va, y_tr, y_va = train_test_split(X_base, y, test_size=0.2, random_state=2020, stratify=y)
    Xe_tr, Xe_va = X_ext.loc[Xb_tr.index], X_ext.loc[Xb_va.index]

    # Baseline search
    _, best_params_base, best_auc_base = tune_lgbm_random_search(
        Xb_tr.values, y_tr, Xb_va.values, y_va, 
        n_iter=n_iter, random_state=2021, desc="Baseline (Agg only)"
    )

    # Extended search
    _, best_params_ext, best_auc_ext = tune_lgbm_random_search(
        Xe_tr.values, y_tr, Xe_va.values, y_va, 
        n_iter=n_iter, random_state=2022, desc="Extended (Agg + Custom)"
    )

    print(f"\nResults - Base AUC: {best_auc_base:.6f} | Ext AUC: {best_auc_ext:.6f} | Gain: {best_auc_ext - best_auc_base:.6f}")
    return best_params_base, best_auc_base, best_params_ext, best_auc_ext

def train_and_save_two_submissions(apps_all_train, apps_all_test, extra_feature_cols, best_params_base, best_params_ext):
    y = apps_all_train["TARGET"].values
    X_full_train = apps_all_train.drop(["SK_ID_CURR", "TARGET"], axis=1)
    X_full_test = apps_all_test.drop(["SK_ID_CURR"], axis=1)
    
    base_cols = [c for c in X_full_train.columns if c not in set(extra_feature_cols)]
    
    # Train Baseline
    print("\nTraining Baseline Model...")
    clf_base = LGBMClassifier(**best_params_base)
    clf_base.fit(X_full_train[base_cols].values, y, eval_metric="auc") # No validation set for final sub
    
    sub_base = pd.DataFrame({"SK_ID_CURR": apps_all_test["SK_ID_CURR"], "TARGET": clf_base.predict_proba(X_full_test[base_cols].values)[:, 1]})
    sub_base.to_csv(os.path.join(default_dir, "submission_base.csv"), index=False)

    # Train Extended
    print("Training Extended Model...")
    clf_ext = LGBMClassifier(**best_params_ext)
    clf_ext.fit(X_full_train.values, y, eval_metric="auc")
    
    sub_ext = pd.DataFrame({"SK_ID_CURR": apps_all_test["SK_ID_CURR"], "TARGET": clf_ext.predict_proba(X_full_test.values)[:, 1]})
    sub_ext.to_csv(os.path.join(default_dir, "submission_ext.csv"), index=False)
    print("Submissions saved.")

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------

def main():
    # 1. Prepare custom feature set
    build_my_features()

    # 2. Load and Aggregate
    apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = get_dataset()
    apps_all = get_apps_all_with_all_agg(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
    
    del apps, prev, bureau, bureau_bal, pos_bal, install, card_bal
    gc.collect()

    # 3. Encode and Split
    apps_all = get_apps_all_encoded(apps_all)
    train_df = apps_all[apps_all["TARGET"].notnull()]
    test_df = apps_all[apps_all["TARGET"].isnull()].drop("TARGET", axis=1)

    # 4. Identify custom columns
    my_feat_cols = pd.read_csv(os.path.join(default_dir, "my_features_train.csv"), nrows=1).columns.tolist()
    extra_feature_cols = [c for c in my_feat_cols if c != "SK_ID_CURR"]

    # 5. Tune
    best_params_base, _, best_params_ext, _ = tune_lgbm_separate_base_and_ext(
        train_df, extra_feature_cols, n_iter=24
    )

    # 6. Final Submission
    train_and_save_two_submissions(train_df, test_df, extra_feature_cols, best_params_base, best_params_ext)

if __name__ == "__main__":
    main()