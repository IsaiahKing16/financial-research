# Optuna for walk-forward financial prediction tuning

**Optuna is the best general-purpose choice for this pipeline.** Its native handling of mixed parameter types, cross-validation-aware pruning, and lightweight footprint make it well-suited for a 6-fold walk-forward setup with 3–5 hyperparameters and ~2-minute trials. With **100 TPE-guided trials (~3.3 hours)**, you can expect results equivalent to 200–500 random search trials, and the WilcoxonPruner can cut total fold evaluations roughly in half by killing hopeless trials after 2 folds. The only serious alternative worth considering is Meta's Ax for its superior sample efficiency in very low-budget regimes, but Optuna's simplicity, active maintenance (v4.8.0, March 2026), and rich ecosystem make it the pragmatic default.

---

## Study definition and objective function architecture

An Optuna study wraps a search process around an objective function that takes a `Trial` object, samples hyperparameters, and returns a scalar metric. For maximizing mean Brier Skill Score across 6 walk-forward folds:

```python
import optuna, numpy as np

def objective(trial):
    max_distance = trial.suggest_float("max_distance", 0.5, 1.0)
    top_k = trial.suggest_int("top_k", 5, 50)
    confidence_threshold = trial.suggest_float("confidence_threshold", 0.55, 0.75)

    bss_scores = []
    for fold in range(6):
        bss = run_walk_forward_fold(fold, max_distance, top_k, confidence_threshold)
        bss_scores.append(bss)
        trial.report(bss, step=fold)          # intermediate value for pruning
        if trial.should_prune():
            return np.mean(bss_scores)         # return partial mean, not TrialPruned
    return np.mean(bss_scores)

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
    pruner=optuna.pruners.WilcoxonPruner(p_threshold=0.1),
    storage="sqlite:///walkforward.db",
    study_name="bss_opt",
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)
```

Several best practices matter for expensive evaluations. First, **use a callable class** instead of a closure when the objective needs access to preloaded data or fold definitions—this avoids serialization issues and keeps the function testable. Second, store per-trial metadata via `trial.set_user_attr("fold_scores", bss_scores)` so you can later analyze variance across folds without re-running anything. Third, note that TPESampler defaults to **10 random startup trials** (`n_startup_trials=10`) before Bayesian-guided sampling begins; at 2 minutes per trial, that's 20 minutes of random exploration, which is reasonable for 3–5 parameters. Reducing to 5 is possible but risks under-exploring the space before the surrogate model kicks in.

One subtle but important detail: when a trial is pruned, returning the partial mean BSS (rather than raising `optuna.TrialPruned()`) gives TPE usable signal from that trial. The official documentation notes that "some samplers, including the default TPESampler, currently cannot utilize the information of pruned trials effectively," so returning a value is the recommended workaround.

---

## TPE dominates CMA-ES for mixed discrete/continuous spaces

For a parameter space mixing continuous floats (`max_distance`, `confidence_threshold`), discrete integers (`top_k`), and potentially categorical choices, **TPESampler is the clear winner**. TPE models each parameter type natively: truncated Gaussian mixtures for continuous variables, re-weighted categorical distributions for categoricals, and appropriate rounding for integers. With `multivariate=True`, it jointly models parameter interactions rather than treating each dimension independently.

CMA-ES, by contrast, **does not support categorical parameters at all**—it falls back to a RandomSampler for any categorical dimension. It handles integers via "CMA-ES with Margin" (`with_margin=True`, available since v3.1), which maintains marginal probabilities to prevent the search from stalling on discrete boundaries, but this adds complexity for modest gain. More critically, CMA-ES typically requires a population size of roughly 4 + 3·ln(d) individuals per generation (about 9–10 for 5 dimensions), meaning it needs more evaluations before meaningful adaptation begins.

The literature supports TPE for small budgets. Loshchilov & Hutter (2016) found that CMA-ES was "competitive only after about 200 evaluations" on a 19-dimensional DNN tuning problem, while TPE performed well throughout. For **50–200 trials in 3–5 dimensions, TPE converges faster and more reliably**. CMA-ES becomes attractive only for purely continuous spaces with larger budgets (>200 trials) where its covariance adaptation can exploit correlations between parameters.

One newer option worth noting: **CatCMA** (available via OptunaHub, published 2024) extends CMA-ES to handle categorical variables natively, but it remains experimental. Optuna also offers an **AutoSampler** via OptunaHub that automatically selects the best algorithm based on problem characteristics—a reasonable choice if you want to avoid manual sampler selection entirely.

---

## WilcoxonPruner is purpose-built for fold-based evaluation

Optuna's pruning system can absolutely stop trials early based on intermediate fold results, and it offers a pruner specifically designed for this scenario. The mechanism is straightforward: after each fold, call `trial.report(bss, step=fold_idx)` and then check `trial.should_prune()`. If the pruner determines the trial is unpromising, you short-circuit the remaining folds.

The **WilcoxonPruner** (added in v3.6.0, still marked experimental) is the ideal choice for cross-validation. Unlike MedianPruner or HyperbandPruner—which assume intermediate values form a learning curve where step N+1 improves on step N—WilcoxonPruner assumes intermediate values are **statistically independent observations** of the same quantity, exactly the semantics of cross-validation folds. It applies a Wilcoxon signed-rank test comparing the current trial's fold scores against the best completed trial's fold scores, pruning when the p-value falls below `p_threshold`.

In Optuna's own benchmarks, WilcoxonPruner **reduced total instance evaluations by roughly 50%**: from 2,500 evaluations (50 instances × 50 trials) to about 1,023. For your 6-fold setup, this means a pruned trial might run only 2–3 folds (~40–60 seconds) instead of all 6 (~120 seconds). With `p_threshold=0.1` and `n_startup_steps=2`, a trial showing BSS ≪ 0 on folds 1 and 2 will almost certainly be pruned before fold 3.

The alternative pruners have different strengths:

- **MedianPruner** prunes if a trial's intermediate value falls below the median of completed trials at the same step. Simple and robust, but assumes learning-curve semantics. Set `n_warmup_steps=1` to allow at least 2 folds before pruning.
- **HyperbandPruner** combines successive halving with multiple budget brackets. Optuna's benchmarks found it pairs best with TPESampler. Set `max_resource=6` (number of folds).
- **SuccessiveHalvingPruner** aggressively eliminates the bottom fraction at each "rung." Works but is less statistically principled than WilcoxonPruner for CV.

One important **gotcha**: pruning is currently supported only for **single-objective** studies. If you need multi-objective optimization with pruning, you must either scalarize your objectives into a single metric or handle early stopping manually within the objective function.

---

## Multi-objective optimization trades pruning for Pareto analysis

Optuna supports multi-objective optimization through `optuna.create_study(directions=["maximize", "minimize"])`. For simultaneously maximizing BSS and minimizing signal suppression rate (percentage of HOLD predictions), the setup is:

```python
study = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=optuna.samplers.NSGAIISampler(population_size=50, seed=42),
)
study.set_metric_names(["BSS", "suppression_rate"])

def objective(trial):
    params = suggest_params(trial)
    bss, suppression_rate = evaluate(params)
    return bss, suppression_rate

study.optimize(objective, n_trials=200)
pareto_trials = study.best_trials  # Pareto-optimal set
```

**NSGAIISampler** (NSGA-II) is the default and most battle-tested multi-objective sampler. It uses non-dominated sorting with crowding distance to maintain diversity across the Pareto front. For two objectives, it works well with `population_size=50`. Optuna also provides **NSGAIIISampler** for three or more objectives (using reference-point-based selection) and **TPESampler** now supports multi-objective since v4.0.0 with improved speed. Additional multi-objective samplers available through OptunaHub include SPEA-II, HypE, and MO-CMA-ES.

Accessing the Pareto front is straightforward: `study.best_trials` returns all non-dominated trials. Note that `study.best_trial` (singular) raises a `RuntimeError` for multi-objective studies—you must work with the Pareto set directly. Optuna provides `plot_pareto_front(study, target_names=["BSS", "suppression_rate"])` for visualization.

The practical trade-off is significant: **you lose pruning in multi-objective mode**. Each trial must run all 6 folds. A pragmatic alternative is to run single-objective optimization maximizing BSS with pruning enabled, then filter the results by acceptable suppression rates. This often gives better results per wall-clock hour for problems where one objective clearly dominates.

---

## Visualization reveals which hyperparameters actually matter

Optuna's visualization suite is genuinely useful for understanding parameter sensitivity, not just monitoring progress. The two most actionable tools are **parameter importance plots** and **parallel coordinate plots**.

Parameter importance uses **fANOVA** (functional ANOVA, Hutter et al. 2014) by default: it fits a random forest to predict objective values from parameter configurations, then decomposes the variance to quantify each parameter's contribution. This tells you which parameters cause the most variation in BSS—critical for deciding whether `top_k` deserves a wider search range or whether `confidence_threshold` barely matters. A newer alternative, **PedANOVA** (added v3.6), answers a more targeted question: "which parameters were important to achieve the top 10% of results?" This is often more actionable because fANOVA can attribute high importance to parameters that *damage* performance when set poorly, even if they don't differentiate good from great.

**optuna-dashboard** (v0.20.0) provides a real-time web interface launched with `optuna-dashboard sqlite:///walkforward.db`. It shows optimization history, parameter importances, hyperparameter relationships, and study management—all without writing plotting code. It's also available as a VS Code extension (right-click your SQLite file), a JupyterLab extension, and even a browser-only version using SQLite Wasm. On Windows, it falls back to the wsgiref server instead of gunicorn, which works fine for local development. The v0.20.0 release added LLM-powered features for natural-language trial filtering and automatic chart generation.

The `plot_contour` function deserves special mention for financial pipelines: it visualizes how pairs of parameters interact, revealing whether `max_distance` and `confidence_threshold` have a compensatory relationship (high distance requiring higher confidence, for instance). Both Plotly (interactive, default) and Matplotlib (publication-quality) backends are available.

---

## SQLite storage makes optimization resumable with one flag

Persistent storage is one of Optuna's most practical features for expensive optimization. Using `storage="sqlite:///study.db"` with `load_if_exists=True` means you can interrupt the process (Ctrl+C cleanly fails the current trial for `n_jobs=1`), close your laptop, and resume later—all completed trials are preserved, and TPE reconstructs its surrogate model from stored history.

The critical **caveat** is that storage does not serialize sampler or pruner state. When resuming, TPE rebuilds from trial history, which is functionally equivalent but not bit-for-bit identical to the interrupted state. For exact reproducibility, pickle the sampler separately. For most practical purposes, the rebuilt state is good enough.

SQLite works well for **single-process sequential optimization**—your ~2-minute/trial workflow. For parallel optimization, SQLite's locking limitations make it unsuitable; the Optuna documentation explicitly warns against it. **JournalStorage** (`JournalFileStorage("optuna-journal.log")`) is the file-based alternative for parallel work, and MySQL/PostgreSQL via `RDBStorage` is recommended for multi-node setups. On Windows specifically, JournalStorage requires `JournalFileOpenLock` to avoid privilege errors.

For long-running studies, configure heartbeat monitoring: `RDBStorage(..., heartbeat_interval=60, grace_period=120, failed_trial_callback=RetryFailedTrialCallback(max_retry=3))`. This detects crashed trials and retries them automatically—valuable when a single trial costs 2 minutes and crashing at trial 95 of 100 would be painful.

---

## Hyperopt is legacy, scikit-optimize is dead, Ax is the real alternative

**Hyperopt** has not released a new version in over 3 years (last release: 0.2.7, ~2021–2022). Databricks has removed it from Runtime ML after version 16.4 LTS, recommending Optuna or Ray Tune instead. It lacks multi-objective optimization, pruning, and its define-and-run API makes conditional parameter spaces awkward. Its TPE implementation is the original Bergstra et al. version, while Optuna's is a modernized re-implementation with improvements. Hyperopt should not be chosen for new projects.

**scikit-optimize** is worse: the repository was **archived on February 28, 2024** and is read-only. It is broken with modern NumPy (2.0+) and scikit-learn (≥1.3) due to deprecated API usage (`np.int` removed). A community fork exists with minimal activity. Do not use it.

**Ax (Meta)** is the only serious competitor and deserves genuine consideration. Built on BoTorch/GPyTorch, it uses Gaussian Process surrogates with proper acquisition functions (Expected Improvement, q-Expected Hypervolume Improvement). A systematic benchmark by Kégl found Ax among the top 3 HPO engines, particularly strong on "extra small budgets." For your scenario of 50–200 trials with expensive evaluations, Ax's GP-based approach is theoretically more sample-efficient than TPE—GPs provide calibrated uncertainty estimates that TPE lacks. The **trade-offs** are substantial, however: Ax requires PyTorch as a dependency, has a steeper learning curve with multiple API levels, scales as O(n³) per trial (becoming slow beyond ~500 trials), and has a smaller community (~2,500 GitHub stars vs Optuna's ~13,600). A Cloudera Labs evaluation concluded: "We'd actually recommend using Optuna for this kind of problem." Ax is best reserved for cases where every trial is extremely expensive (hours, not minutes) and you need maximum sample efficiency from <50 trials.

Other options to know about: **Ray Tune** is an orchestration framework that can use Optuna as its search backend, useful for distributing trials across clusters but overkill for single-machine work. **SMAC3** (v2.x) is actively maintained by the AutoML Freiburg/Hannover groups and competitive in benchmarks but has a smaller community and requires a C++ dependency (swig).

---

## TPE needs roughly 50–100 trials to decisively beat grid search in 3–5 dimensions

The sample efficiency question has strong empirical answers. Bergstra & Bengio (2012) established that **random search outperforms grid search** for hyperparameter tuning because grid search wastes evaluations on unimportant parameter dimensions—with budget B and N parameters, grid search tests only B^(1/N) distinct values per parameter. TPE then outperforms random search by concentrating evaluations in promising regions.

For 3–5 parameters, the convergence timeline is well-characterized. TPE typically begins outperforming random search after **20–50 trials** (including the 10 random startup trials). The original TPE paper showed clear superiority within 200 trials on a *32-dimensional* space; for 3–5 dimensions, the budget is proportionally generous. A practical rule of thumb is **10–20× the number of dimensions** for guided trials: for 5 parameters, that's 50–100 guided trials plus 10 random startup trials, totaling 60–110 trials.

The NeurIPS 2020 Black-Box Optimization Challenge (Turner, Eriksson, McCourt et al., 2021) provided decisive evidence: across 6 ML model types on real datasets with 3D search spaces, **all top-5 competition winners used Bayesian optimization**, and BO methods consistently outperformed random search. One practitioner comparison found that 100 Optuna trials matched the optimum found by an 810-point grid search, achieving it by trial 67.

For your specific scenario—**100 trials (~3.3 hours) is a strong budget for 3–5 parameters**. This should substantially outperform any manual grid search. If time permits, 200 trials (~6.7 hours) approaches near-optimal for smooth 5D spaces. With WilcoxonPruner reducing fold evaluations by ~50%, the effective wall-clock time drops further. Setting `multivariate=True` on TPESampler is important: it enables joint modeling of parameter interactions, which Falkner et al. (2018) showed outperforms independent per-parameter modeling.

---

## Platform compatibility and practical deployment

**Windows 11**: Optuna core works reliably. Officially supported since v3.2.0 with CI testing on Windows. The only Windows-specific issues are peripheral: optuna-dashboard falls back to wsgiref instead of gunicorn (functional but development-grade), and JournalStorage requires `JournalFileOpenLock` to avoid privilege errors. SQLite storage works without issues.

**Python 3.12+**: Fully supported. Optuna 4.8.0's `pyproject.toml` lists Python 3.9 through 3.14, with CI testing across all versions. Python 3.12 support was tracked as GitHub issue #5000 and fully resolved. The `optuna-integration` package supports Python 3.9–3.13.

**Memory**: For fewer than 1,000 trials, memory is a non-issue. Optuna stores only hyperparameters and objective values, not model artifacts. With SQLite backend, trial data is persisted to disk, though a `_CachedStorage` wrapper keeps recent trials in memory for speed. Enable `gc_after_trial=True` in `study.optimize()` if your objective function allocates significant memory per trial. For your ~100-trial scenario, expect negligible overhead.

**Parallel execution on a single machine**: `study.optimize(objective, n_trials=100, n_jobs=4)` runs trials across threads. For CPU-bound objectives that release the GIL (most scikit-learn and NumPy operations), this provides real speedup. Enable `TPESampler(constant_liar=True)` to prevent parallel workers from sampling near-identical configurations—without it, workers running simultaneously may propose similar parameter sets because TPE doesn't account for in-progress trials. The ask-and-tell interface (`study.ask()` / `study.tell()`) enables batch optimization patterns for more control.

---

## Conclusion

Optuna is the right tool for this pipeline. The recommended configuration—**TPESampler with `multivariate=True`, WilcoxonPruner, and SQLite storage**—directly addresses the walk-forward evaluation pattern. Three insights stand out beyond the obvious feature comparison. First, returning partial means from pruned trials instead of raising `TrialPruned()` is a non-obvious but important optimization that feeds information back to TPE. Second, the choice between single-objective-with-pruning and multi-objective-without-pruning is a genuine architectural decision: for BSS + suppression rate, starting with single-objective BSS optimization and filtering by suppression rate post-hoc is likely faster than running unpruned multi-objective trials. Third, the parameter importance analysis (especially PedANOVA over fANOVA) often reveals that one or two parameters dominate, letting you fix the unimportant ones and narrow the search space for subsequent rounds—a meta-optimization strategy that compounds the efficiency of Bayesian search.