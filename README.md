# FDS Challenge 2026: Pokémon Battle Prediction

Code and report for our submissions to the FDS 2026 Kaggle competition at Sapienza:
predicting the winner of a Gen 2 OU battle from its first 30 turns.

| # | Submission file | Model | Private | Public | 5-fold CV |
|---|---|---|---|---|---|
| 1 | `submission_blend_lr_hgb.csv` | Logistic regression + HistGradientBoosting on feature set B | 0.68200 | 0.69333 | 0.685 |
| 2 | `submission_blend.csv` | First version of the same blend | 0.68114 | 0.69933 | 0.689 |
| 3 | `submission_merged_ensemble.csv` | The blend plus LightGBM and XGBoost on feature set A | 0.68085 | 0.69733 | 0.688 |

## Contents

- `pokemon_battle_prediction.ipynb`: the Kaggle notebook. It clones this repository, builds the
  features, runs the cross-validation and the evaluation, and regenerates the three submissions.
- `pokefds/`: the code.
  - `features_b.py`: feature set B (233 features).
  - `features.py`, `features_ext.py`: feature set A (291 features).
  - `models.py`: model definitions and the cross-validation helper.
  - `submissions.py`: the recipes of the three final submissions.
  - `pinned.py`: runs a recipe with the library versions it was originally made with.
  - `evaluation.py`: metric tables and figures.
- `requirements/`: the library versions used for each submission. The versions change a few
  percent of the predictions, so they are pinned.
- `submitted/`: the files as we uploaded them to Kaggle, to compare with the regenerated ones.
- `report/`: the report (`report.pdf`) and its LaTeX source.

## Running it

On Kaggle: import the notebook into the competition, attach the competition data, turn Internet on
and run all cells.

Locally: put the competition files in `data/` (or in a folder named
`fds-pokemon-battles-prediction-2026` next to this repository) and run the notebook from this folder.
A single submission can be generated with

    python -m pokefds.submissions blend_lr_hgb --out output

which uses the libraries of the current environment.
