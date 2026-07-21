# Running Task 1 training on Kaggle

Code lives at `github.com/aaronjji/rk8391-devbox` (public, deliberately named to be non-descriptive). Since Kaggle
notebooks are the only interface (no SSH), this has to be run manually --
these are the setup steps, then `train_notebook.py` has the actual cells.

## One-time setup

1. **GitHub token**: go to github.com/settings/tokens -> Generate new token
   (classic) -> scope `repo` -> copy it. This lets the private repo get
   cloned from inside the Kaggle notebook.
2. **New Kaggle Notebook**: kaggle.com/code -> New Notebook.
   - Settings (right panel) -> Accelerator: **GPU T4 x2** (or P100).
   - Settings -> Internet: **On**.
3. **Add the GitHub token as a Kaggle Secret**: Add-ons -> Secrets -> New
   Secret -> name it `GH_TOKEN`, paste the token, attach it to this notebook.
4. **Upload the dataset privately**: kaggle.com/datasets -> New Dataset ->
   upload your local `data/raw/` folder -> set visibility
   to **Private** (dataset usage terms prohibit public re-hosting). Then in the
   notebook, Add Input -> select that dataset. Note the slug Kaggle assigns
   it (shown in the Add Input panel) -- you'll need it for Cell 3.
5. Copy each `# %%` block from `train_notebook.py` into its own cell in the
   Kaggle notebook, in order, editing `YOUR-DATASET-SLUG` in Cell 3 to match
   step 4.

## Running

Run cells 1-5 top to bottom. Cell 5 trains fold 0 for up to `max-seconds`
(default 30000s ≈ 8.3h, under Kaggle's session cap). To train the other
folds, change `FOLD` in Cell 5 and re-run just that cell.

## Resuming across sessions

Kaggle sessions don't run forever, and the free quota is 30 GPU-hours/week:

1. When a session is about to end (or `--max-seconds` triggers), the script
   checkpoints to `runs/task1/fold{N}/latest.pth` automatically.
2. **Save Version** (commit) the notebook -- this persists `/kaggle/working`
   as a versioned output.
3. Next session: Add Input -> pick the previous version's output -> set
   `PREV_CHECKPOINT` in Cell 4 to its path -> re-run cells 1-5 (Cell 5 already
   passes `--resume`, so it continues from the restored checkpoint).

## Getting results back

Once a fold's training looks converged (watch `runs/task1/foldN/train_log.csv`
for loss plateauing), download `runs/task1/foldN/final.pth` (or the latest
epoch checkpoint) from the notebook's output files, and either:
- run `src/predict_task1.py` + `src/run_task3.py` + `src/format_submission.py`
  right there in the Kaggle notebook (same environment, GPU available for
  inference too), or
- pull the checkpoint back down locally and run inference on the laptop
  (inference is much cheaper than training -- no backward pass).
