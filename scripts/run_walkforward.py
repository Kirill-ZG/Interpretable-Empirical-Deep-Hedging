"""Run empirical walk-forward training and testing experiments.

For each target year, this script builds isolated train/validation/test data
folders, trains the neural hedging agent, selects the best validation
checkpoint, and evaluates it on the corresponding out-of-sample year.
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
import subprocess
import shutil
import multiprocessing
import random

KAPPA = 1.0
REWARD_EXPONENT = 1.0
TRANSACTION_COST = 0.0
NUM_EPISODES = 20000
VALIDATION_INTERVAL = 1000

START_TRAIN_YEAR = 2010      
FIRST_TEST_YEAR = 2015
FINAL_TEST_YEAR = 2023       

MAX_CONCURRENT_WORKERS = 3
MODEL_PREFIX = "final_WF_exp1_k1_test"
BASE_SEED = 20260427


def set_worker_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def create_window_datasets(target_test_year):
    """Build train, validation, and test CSVs for one walk-forward year."""
    print(f"[{target_test_year}] Constructing isolated datasets...")
    
    window_data_dir = f"data_wf_{target_test_year}"
    os.makedirs(window_data_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join("cleaned_data", "*.parquet")))
    if not all_files:
        raise FileNotFoundError("No parquet files found in 'cleaned_data/'.")
        
    df_list = [pd.read_parquet(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    
    df = df.dropna(subset=['quote_date', 'underlying_last', 'expire_date', 'strike', 'c_bid', 'c_ask', 'risk_free_rate'])
    df = df[(df['c_bid'] > 0) & (df['c_ask'] > df['c_bid'])]
    intrinsic_value = df['underlying_last'] - df['strike']
    df = df[df['c_bid'] >= intrinsic_value]
    df = df[df['dte'] > 0]
    
    treasury_df = df[['quote_date', 'risk_free_rate']].drop_duplicates(subset=['quote_date'])
    treasury_df = treasury_df.rename(columns={'quote_date': 'Date', 'risk_free_rate': '1y'})
    treasury_df = treasury_df.sort_values('Date')
    treasury_df.to_csv(os.path.join(window_data_dir, '1yr_treasury.csv'), index=False)

    df = df.rename(columns={'quote_date': 'quote_datetime', 'expire_date': 'expiration', 'c_bid': 'bid', 'c_ask': 'ask'})
    df['underlying_bid'] = df['underlying_last']
    df['underlying_ask'] = df['underlying_last']
    df['ticker'] = 'SPX'
    
    keep_cols = ['quote_datetime', 'expiration', 'strike', 'underlying_bid', 'underlying_ask', 'bid', 'ask', 'ticker']
    df = df[keep_cols]
    df['quote_datetime'] = df['quote_datetime'].astype(str).str.slice(0, 10)
    df['expiration'] = df['expiration'].astype(str).str.slice(0, 10)
    df['year'] = df['quote_datetime'].str.slice(0, 4).astype(int)

    train_years = list(range(START_TRAIN_YEAR, target_test_year - 1))
    val_years = [target_test_year - 1]
    test_years = [target_test_year]
    
    df_train = df[df['year'].isin(train_years)].copy()
    df_val   = df[df['year'].isin(val_years)].copy()
    df_test  = df[df['year'].isin(test_years)].copy()
    
    df_train.drop(columns=['year'], inplace=True)
    df_val.drop(columns=['year'], inplace=True)
    df_test.drop(columns=['year'], inplace=True)

    def process_and_save(dataset, filename):
        dataset['option_id'] = dataset['expiration'] + "_" + dataset['strike'].astype(str)
        dataset = dataset.sort_values(['option_id', 'quote_datetime']).reset_index(drop=True)
        dataset['nbr_next_steps'] = dataset.groupby('option_id').cumcount(ascending=False)
        dataset = dataset.drop(columns=['option_id'])
        dataset.to_csv(os.path.join(window_data_dir, filename), index=False)

    process_and_save(df_train, 'train.csv')
    process_and_save(df_val, 'validation.csv')
    process_and_save(df_test, 'test.csv')

    if os.path.exists("data/heston_params.csv"):
        shutil.copy("data/heston_params.csv", os.path.join(window_data_dir, "heston_params.csv"))
    else:
        pd.DataFrame({'date': [], 'v0': [], 'kappa': [], 'theta': [], 'sigma': [], 'rho': []}).to_csv(
            os.path.join(window_data_dir, "heston_params.csv"), index=False
        )
        
    return window_data_dir

def create_settings_json(target_test_year):
    model_name = f"{MODEL_PREFIX}{target_test_year}"
    seed = BASE_SEED + target_test_year
    
    settings = {
        "process"       : "Real",
        "transaction_cost"  : TRANSACTION_COST,
        "kappa"         : KAPPA,
        "reward_exponent"   : REWARD_EXPONENT,
        "n_steps"       : 21,
        "D"             : 1,
        "sim_test_runs" : 100,
        "showcase_every": 1000,
        "validation_interval" : VALIDATION_INTERVAL,
        "validation_limit"  : 25, 
        "num_episodes"   : NUM_EPISODES,
        "min_noise"     : 0.2,
        "max_noise"     : 0.7,
        "noise_reward_dividor" : 150,
        "q" 			: 0.018,
        "SIGMA"         : 0.25,
        "batch_size"    : 1000,
        "actor_lr"      : 1e-4,
        "critic_lr"     : 1e-4,
        "tau"           : 0.001,
        "discount"      : 1.0,
        "policy_noise"  : 0.2,
        "policy_noise_max"  : 0.5,
        "policy_freq"   : 2,
        "actor_nn"      : 250,
        "critic_nn"     : 250,
        "lrelu_alpha"   : 0.05,

        "seed"          : seed
    }
    
    settings_dir = os.path.join("settings", MODEL_PREFIX)
    os.makedirs(settings_dir, exist_ok=True)
    settings_path = os.path.join(settings_dir, f"{model_name}.json")
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=4)
        
    return model_name

def worker_task(test_year):
    """Run training, validation, and testing for one walk-forward year."""
    print(f"\n--- [START] Walk-Forward Worker for Test Year {test_year} ---")
    
    try:
        worker_seed = BASE_SEED + test_year
        set_worker_seed(worker_seed)

        data_dir = create_window_datasets(test_year)
        model_name = create_settings_json(test_year)
        
        env_vars = os.environ.copy()
        env_vars["DATA_DIR"] = data_dir
        env_vars["PYTHONHASHSEED"] = str(worker_seed)
        env_vars["PYTHONDONTWRITEBYTECODE"] = "1"
        env_vars["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + env_vars.get("PYTHONPATH", "")

        print(f"[{test_year}] Training {model_name}...")
        with open(f"logs/train_{test_year}.log", "w") as log_out:
            subprocess.run(
                [sys.executable, "-m", "empirical_deep_hedging.main", "--settings", model_name],
                env=env_vars, stdout=log_out, stderr=subprocess.STDOUT, check=True
            )
        
        print(f"[{test_year}] Validating to find best checkpoint...")
        val_result = subprocess.run(
            [sys.executable, "-m", "empirical_deep_hedging.testing", "--validate", "--model", model_name],
            env=env_vars, capture_output=True, text=True, check=True
        )
        
        with open(f"logs/val_{test_year}.log", "w") as log_out:
            log_out.write(val_result.stdout)
            
        best_checkpoint = None
        for line in val_result.stdout.split('\n'):
            if line.startswith("Best:"):
                best_checkpoint = line.split(',')[0].replace("Best: ", "").strip()
                
        if not best_checkpoint:
            print(f"[{test_year}] Warning: Could not parse validation output. Using epoch {NUM_EPISODES}.")
            best_checkpoint = f"{model_name}_{NUM_EPISODES}"
            
        print(f"[{test_year}] Selected Checkpoint: {best_checkpoint}")

        print(f"[{test_year}] Testing Out-of-Sample...")
        with open(f"logs/test_{test_year}.log", "w") as log_out:
            subprocess.run(
                [sys.executable, "-m", "empirical_deep_hedging.testing", "--test", "--model", best_checkpoint],
                env=env_vars, stdout=log_out, stderr=subprocess.STDOUT, check=True
            )

        print(f"[{test_year}] Success. Cleaning up temp data...")
        shutil.rmtree(data_dir)
        
        return f"Test {test_year} Completed Successfully."

    except Exception as e:
        return f"Test {test_year} FAILED: {e}"

if __name__ == "__main__":
    print("="*75)
    print(" EXPANDING WINDOW WALK-FORWARD BACKTEST (PARALLEL EXECUTION)")
    print("="*75)
    print(f"Risk Aversion (Kappa) : {KAPPA}")
    print(f"Reward Exponent       : {REWARD_EXPONENT}")
    print(f"Transaction Cost      : {TRANSACTION_COST}")
    print(f"Test Years            : {FIRST_TEST_YEAR} to {FINAL_TEST_YEAR}")
    print(f"Concurrent Workers    : {MAX_CONCURRENT_WORKERS}")
    print("="*75)
    
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    years_to_test = list(range(FIRST_TEST_YEAR, FINAL_TEST_YEAR + 1))
    
    with multiprocessing.Pool(processes=MAX_CONCURRENT_WORKERS) as pool:
        results = pool.map(worker_task, years_to_test)
        
    print("\n" + "="*75)
    print(" WALK-FORWARD BACKTEST COMPLETE ")
    print("="*75)
    for res in results:
        print(res)
