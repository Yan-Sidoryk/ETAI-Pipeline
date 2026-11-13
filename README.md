# Predictive Pipeline -- ETAI -- 20222004

**Task:** predict whether a person earns more than 50,000 USD a year (`income_over_50k`, 1 = yes, about 24% of people) from 1994 US Census data (UCI *Adult*, subsampled and extended for the course). `data/adult_train.csv` has 6,000 labelled people; `data/adult_new.csv` has 2,000 new people the final model predicts. Column descriptions: `data/README.md`.

The pipeline was built on ProPublica's COMPAS recidivism data (weeks 2-8) and moved to Adult income in week 9; COMPAS has since been removed. Its history is in git (up to the "week 8 submission" commit).

## Running the pipeline

With the environment active (see "Environment setup"), from the project's root folder:
```bash
python main.py                 # uses config.yaml
python main.py other.yaml      # any other config file
```

It loads and cleans the data, locks a stratified 20% test set away, cross-validates the whole pipeline (preprocessing -> feature selection -> model) on the development set with 8 metrics and the no-model floor, tunes it with Optuna under nested CV (if `tuning.enabled`), prints a fairness check (false positive rate by `sex`), refits on all development rows and predicts `data/adult_new.csv`.

Outputs:
- `results/run_<timestamp>.txt` -- full report of each run (per-fold tables, tuning, confusion matrix, fairness, features kept); `results/runs.csv` -- scoreboard of all runs. Not tracked in git.
- `predictions/adult_new_predictions.csv` -- `id`, probability and 0/1 prediction for all 2,000 new people. Tracked in git.

Tuning with feature selection takes several minutes; set `tuning.enabled: false` in `config.yaml` for quick runs.

## Project structure

```
.
├── main.py                  # entry point: python main.py [config]  (default config.yaml)
├── config.yaml              # all settings: data, cleaning rules, preprocessing, model, selection, tuning, output
├── requirements.txt
├── src/
│   ├── data.py              # loading
│   ├── preprocessing.py     # rule-based cleaning, feature engineering, feature/target split, leak-safe ColumnTransformer, dev/test split
│   ├── model.py             # model construction (incl. XGBoost) + VoteSelector feature selection + build_pipeline()
│   ├── evaluate.py          # cross-validation on several metrics, out-of-fold probabilities, confusion matrix + fairness check, paired test
│   ├── tuning.py            # Optuna tuning + nested cross-validation
│   ├── predict.py           # predicts label-free rows (data.predict_path) with the refit pipeline
│   └── results.py           # saves each run's report, and its scores to the scoreboard
├── predictions/
│   └── adult_new_predictions.csv
├── results/                 # created automatically, not tracked in git
└── data/
    ├── adult_train.csv      # 6,000 labelled rows
    ├── adult_new.csv        # 2,000 rows to predict
    └── README.md            # column descriptions
```

## Pipeline progress

| Week | Focus | Change |
|------|-------|--------|
| 2-8 | COMPAS recidivism | baseline pipeline, EDA-driven cleaning, leak-safe preprocessing, cross-validation on 8 metrics, fairness check, model zoo (trees, ensembles, boosting, XGBoost), Optuna tuning with nested CV, VoteSelector feature selection, scoreboard + corrected paired t-test |
| 9 | New dataset: Adult income | config chosen on the command line; placeholder cleaning covers every text column; feature engineering chosen in the config; generic fairness report; `src/predict.py` predicts `adult_new.csv`; fixed the target encoder and count-encoded feature names. Then COMPAS removed: Adult is the only task and `config.yaml` the only config. |

## Data decisions

All in `config.yaml`; no column is named in `src/` except in `add_adult_features`.

| Column(s) | Issue | What was done |
|---|---|---|
| `income_over_50k` | target, 24% positive | stratified folds and test split |
| `id` | identifier | not a feature; predictions are saved by `id` |
| `sex` | sensitive attribute | never a feature; audited in the fairness report |
| `race` | second protected attribute | not a feature either: income can be predicted without it, so the model shouldn't use it |
| `fnlwgt` | census sampling weight, not a trait of the person | dropped |
| `education` | one-to-one with `education_num` | dropped, `education_num` keeps the order |
| `workclass`, `occupation`, `native_country` | `?` placeholder (379 / 381 / 86 rows) | turned into missing in cleaning, then mode-imputed |
| `workclass` / `occupation` missing | same people, who earn > 50K half as often (12% vs 25%): MNAR | `workclass_was_missing` flag |
| `native_country` `?`, empty cells in `v01`-`v06` | ~3%, no relation with the target | impute only |
| `age`, `hours_per_week`, `capital_gain`, `capital_loss` | validity rules: age 17-90 (74 valid 17-year-olds), hours 1-99, gains/losses ≥ 0 | violations -> missing (none in these files) |
| `capital_gain` | 92% zero, 99999 is the census top code | kept as real values; log and has-gain features added |
| `v01`-`v25` | undocumented | kept as features, left to feature selection (none survives) |
| duplicates | none | `drop_duplicate_rows` kept for training data; never applied to the rows to predict |

## Moving from COMPAS to Adult: what broke, why, and the fix

| What broke | Why (the hidden COMPAS assumption) | Fix |
|---|---|---|
| no way to run another config | `main.py` always loaded `config.yaml` | `python main.py <config>`, default `config.yaml` |
| `?` survived as a real category in `workclass`, `occupation`, `native_country` (silent: no error) | placeholders were only cleaned in columns listed in `canonical_categories`, and on COMPAS every column with placeholders happened to be listed | `clean_dataset` turns placeholders into NaN in **every** text column |
| `KeyError: 'priors_count'` | feature engineering built COMPAS features unconditionally | `add_adult_features`, chosen by `preprocessing.feature_engineering: adult \| none` |
| `KeyError: 'score_text'` in the fairness report | the fairness report always compared against COMPAS's own score, and said "reoffend" | fairness report uses only the configured sensitive attribute; neutral wording |
| no way to predict new rows | the pipeline only ever evaluated labelled data | `src/predict.py`: same cleaning (no de-duplication), refit pipeline, `id` + probability + 0/1; checks rows out = rows in |
| "Features kept: categorical__1, categorical__2, ..." | the imputer handed `CountEncoder` a bare array, so it named columns by position (on COMPAS no categorical was ever kept, so nobody saw it) | imputer outputs pandas, names survive |
| `encoder: target` crashed | `TargetEncoder(cv=StratifiedKFold(...))` is rejected by current sklearn | `cv=5, shuffle=True, random_state=seed` (still stratified for a binary target) |
| predictions would not be committed | `results/` is gitignored | predictions go to `predictions/` |

## Results

5-fold stratified CV on the 4,800 development rows; the 1,200-row test set is still locked.

| Run | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| targets | ≥ 0.90 | ≥ 0.855 | ≥ 0.66 |
| no model (always "≤ 50K") | 0.50 | 0.76 | 0.00 |

*Comparison runs, untuned, xgboost (`n_estimators 300, learning_rate 0.05, max_depth 4`), selection min_votes 2 unless stated:*

| Run | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| xgboost, count encoder | 0.912 | 0.861 | 0.680 |
| xgboost, **target encoder** | 0.912 | 0.864 | 0.686 |
| xgboost, onehot encoder | 0.905 | 0.853 | 0.655 |
| xgboost, ordinal encoder | 0.908 | 0.855 | 0.659 |
| xgboost, count encoder + `scale_pos_weight` 3.2 | 0.912 | 0.820 | 0.691 |
| xgboost, count encoder, no engineered features | 0.913 | 0.862 | 0.680 |
| xgboost, count encoder, min_votes 1 | 0.912 | 0.862 | 0.679 |
| xgboost, count encoder, no feature selection | 0.906 | 0.860 | 0.667 |
| random forest (300 trees, min leaf 5), target encoder | 0.903 | 0.854 | 0.652 |
| gradient boosting (sklearn defaults), target encoder | 0.911 | 0.862 | 0.676 |

*Final model, the one in `config.yaml`:*

| Run | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| **gradient boosting, target encoder, tuned (30 trials, nested CV, ~4.5 min)** | **0.908 ± 0.014** | **0.856 ± 0.008** | **0.663 ± 0.026** |

Tuned hyperparameters: `n_estimators 131, learning_rate 0.076, max_depth 3, min_samples_leaf 4, subsample 0.90, select__min_votes 1`. xgboost is no longer tuned (too slow); its untuned runs stay above for comparison.

**Features kept by the final model (10 of 41):** age, education_num, capital_gain, capital_loss, hours_per_week, log_capital_gain, capital_net, marital_status, occupation, relationship. **No `v` column** is kept (they only get in with min_votes 1 *and* an untuned model, without improving anything).

**Fairness check** (out-of-fold, development set): false positive rate 0.02 for women vs 0.08 for men. Women are much less often predicted > 50K, which mostly follows the base rate in the 1994 data.

**Predictions** for the 2,000 new people: [`predictions/adult_new_predictions.csv`](predictions/adult_new_predictions.csv) (18.5% predicted > 50K), from the tuned gradient boosting model.

**Reflection.** The hardest assumption to find was the placeholder cleaning. It didn't crash, it just quietly kept `?` as a category, because on COMPAS every column with placeholders happened also to have a spelling map. The same goes for the unreadable feature names: they only showed up once a categorical feature was strong enough to be selected. What reached the targets was mostly good cleaning plus a strong default model. Boosting (xgboost or sklearn's gradient boosting) with feature selection cleared all three targets before any tuning. Tuning didn't add anything here: the honest nested score of the tuned model is slightly below the untuned one, and only just above the accuracy and F1 targets. Class weighting is a clear trade-off: F1 goes up, but accuracy falls below the target.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Environment Troubleshooting

Two Windows issues come up often enough to note here -- if you hit either, this saves you re-diagnosing it from scratch.

**PowerShell blocks the venv activation script, every new terminal.** The `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` line above only fixes it for that one terminal window -- close it and it's back. For a fix that actually sticks across sessions, run this **once**, instead:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
If it still doesn't stick (common on locked-down school/lab machines with a Group Policy that resets it on every logon), skip PowerShell entirely: use **Git Bash** (`source venv/Scripts/activate`) or **cmd.exe** (`venv\Scripts\activate.bat`) instead -- neither is affected by PowerShell's execution policy.

**Windows blocks the terminal/Python from reading or writing files in Documents (or Desktop/Pictures).** Shows up as an "Access is denied" error, or a silent failure to create/update a file, only when the project sits inside one of those folders. Two independent settings can cause this -- check both:
- **Windows Security -> Virus & threat protection -> Manage ransomware protection** -- turn off **Controlled folder access**, or add your terminal/Python/editor to its allowed-apps list.
- **Settings -> Privacy & security -> File system** -- make sure the terminal/Python has access.

## Dataset

See `data/README.md`.
