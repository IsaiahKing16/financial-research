# TurboQuant Integration — Design Specification Addendum

**Version:** 1.0
**Date:** 2026-03-28
**Status:** APPROVED — Ready for Phase 6A implementation
**Parent Spec:** `docs/superpowers/specs/2026-03-26-fppe-full-roadmap-design.md` v1.0

---

## 1. Purpose

This addendum extends the FPPE Full Roadmap Design Specification with the architectural design for integrating Google's TurboQuant vector quantization (arXiv:2504.19874), FAISS scaling infrastructure, and STUMPY matrix profile pattern discovery into the FPPE system.

These capabilities are **additive** — they do not modify the core 10-phase architecture. They add:
- Phase 6A (TurboQuant R&D) — parallel research track
- Phase 6B (Vector Quantization Integration) — conditional on 6A results
- Phase 6, Task 6.5 (FAISS Evaluation) — informational benchmark
- Phase 7, Enhancement 6 (STUMPY) — gated enhancement
- Phase 8, Task 8.7 (Autonomous EOD) — production scheduling
- Phase 11 (Hyper-Scale 5200T+) — post-launch expansion

---

## 2. TurboQuant Architecture within FPPE

### 2.1 How TurboQuant Maps to FPPE's Pipeline

FPPE's 5-stage PatternMatcher pipeline processes vectors as follows:

```
Stage 1: _prepare_features()     →  Raw float64 (N, 8) → Scaled float64 (N, 8)
Stage 2: _build_index()          →  Scaled → Backend.fit() [BallTree or HNSW]
                                     ▲
                                     │ TurboQuant inserts HERE
                                     │ Between scaling and index storage
                                     │
Stage 3: _query_batch()          →  Query (Q, 8) → Backend.kneighbors()
Stage 4: _post_filter()          →  Distance/ticker/sector/regime filtering
Stage 5: _package_results()      →  Platt calibration → signals
```

TurboQuant operates at **Stage 2** — it compresses the scaled feature vectors before they are stored in the index. At query time (Stage 3), queries are full-precision; distance computation is asymmetric (full-precision query vs. decoded-from-quantized database vectors).

### 2.2 Two-Stage Quantization Pipeline

```
Input: X_scaled (N, 8) float64 — from StandardScaler
                │
                ▼
┌─────────────────────────────────────┐
│  Stage 0: Random Orthogonal Rotation │
│  R = QR(randn(8,8))                 │
│  X_rot = X_scaled @ R.T             │
│  Each coordinate → Beta(3.5, 3.5)   │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Stage 1: Lloyd-Max Scalar Quantize  │
│  Per-coordinate: find nearest centroid│
│  b bits/dim → 2^b levels per coord   │
│  Output: indices (N, 8) uint8        │
│  X_hat = dequantize(indices)         │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  Stage 2: QJL Residual Correction    │
│  residual = X_rot - X_hat           │
│  signs = sign(S @ residual.T)       │
│  Output: signs (N, 8) int8          │
│  Provides unbiased IP correction     │
└───────────────┬─────────────────────┘
                │
                ▼
Storage: {indices: uint8(N,8), signs: int8(N,8), R: float64(8,8), codebook: float64(2^b)}
Total: ~2 bytes/vector (vs 32 bytes float32) = 16× compression at d=8
```

### 2.3 Distance Computation (Asymmetric)

At query time, the query vector is full-precision. Distance is computed against decoded vectors:

```python
# Decode training vectors (approximate)
X_hat_rot = codebook[indices]          # (N, 8) float32
X_hat = X_hat_rot @ R                  # inverse rotation

# Euclidean distance: full-precision query vs decoded database
# Uses standard brute-force or FAISS for the distance computation
distances = cdist(X_query, X_hat, metric='euclidean')
```

**Why asymmetric?** The query is a single vector (or small batch) — no memory pressure. Only the database (millions of vectors) needs compression. This matches TurboQuant's design for KV caches: queries are full-precision, cached keys are quantized.

### 2.4 Dimensionality Analysis for d=8

| Property | Value at d=8 | Value at d=200 (paper) | Implication |
|----------|-------------|----------------------|-------------|
| Beta shape | Beta(3.5, 3.5) | Beta(99.5, 99.5) | d=8 is broad; d=200 is near-Gaussian |
| Beta kurtosis | -2/(8+1) ≈ -0.22 | -2/(200+1) ≈ -0.01 | d=8 has heavier tails |
| QJL variance/component | π/(2×8) ≈ 0.196 | π/(2×200) ≈ 0.008 | d=8 has 25× higher noise |
| Theoretical MSE (3-bit) | ~0.03 × (non-ideal factor) | ~0.03 | Factor may be 1.5-3× at d=8 |
| Lloyd-Max optimality | Suboptimal (non-Gaussian) | Near-optimal | Codebook less efficient |

**Realistic expectations:**

| Bits | Estimated Recall@50 (d=8) | Confidence |
|------|--------------------------|------------|
| 1 | 0.60–0.80 | Low — QJL alone is too noisy |
| 2 | 0.80–0.92 | Medium — may be usable for coarse filtering |
| 3 | 0.92–0.98 | Medium-High — promising for FPPE |
| 4 | 0.97–0.999 | High — likely sufficient |
| 5+ | 0.999+ | Very High — near-lossless |

**Decision framework:** The Phase 6A experiment will empirically determine these values. The 0.9990 recall gate is calibrated to ensure BSS does not degrade meaningfully — FPPE's post-filter stage already handles some neighbor noise via distance thresholds.

---

## 3. FAISS Integration Architecture

### 3.1 Index Selection Strategy

```
                          ┌────────────────┐
                          │ How many tickers│
                          └───────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │              │
              ≤ 585T        585T–1500T       > 1500T
                    │             │              │
              ┌─────┴─────┐ ┌────┴─────┐  ┌────┴─────┐
              │ BallTree  │ │ hnswlib  │  │ FAISS    │
              │ (exact)   │ │ (approx) │  │ IVF-Flat │
              └───────────┘ └──────────┘  └──────────┘
              ~260K vectors  ~3.7M vectors  ~26M vectors
              Default        use_hnsw=True  use_faiss=True
```

### 3.2 FAISS IVF-Flat Configuration for d=8

At d=8, FAISS's IVF-Flat (exact within cells) is preferred over IVF-PQ because:
- PQ requires d ≥ 2M (M = number of subquantizers). At d=8, M ≤ 4 → only 2 bits per subvector → heavy quantization loss
- IVF-Flat is exact within each Voronoi cell — recall depends only on nprobe (number of cells searched)

```python
# Recommended configuration for 5200T (~26M vectors, d=8)
n_cells = int(np.sqrt(26_000_000))  # ~5099 cells
nprobe = 32                          # search 32/5099 = 0.6% of cells

# Expected recall: > 0.9999 at nprobe=32 for d=8
# (low-dimensional data has well-separated clusters → few boundary effects)
```

### 3.3 BaseMatcher Interface Compliance

All new matchers (TurboQuantMatcher, FAISSMatcher) implement the existing `BaseMatcher` ABC:

```python
class BaseMatcher(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> None: ...

    @abstractmethod
    def kneighbors(self, X: np.ndarray, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]: ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool: ...

    @abstractmethod
    def get_params(self) -> Dict[str, object]: ...
```

**Return contract (unchanged):**
- `distances`: shape (n_queries, k), **Euclidean** (not squared), sorted ascending
- `indices`: shape (n_queries, k), integer indices into training set

Both FAISS and hnswlib return squared L2 internally — both matchers apply `np.sqrt()` before returning.

---

## 4. STUMPY Integration Architecture

### 4.1 How STUMPY Complements KNN

| Aspect | KNN Fingerprint Matching | STUMPY Matrix Profile |
|--------|--------------------------|----------------------|
| What it matches | Point-in-time return snapshots | Subsequence shape patterns |
| Distance metric | Euclidean on vol-normalized returns | z-normalized Euclidean on raw prices |
| Cross-ticker | Yes (pooled index) | Yes (AB-join) |
| Temporal alignment | None (snapshot at one date) | DTW-like via z-normalization |
| Output | Probability of 7-day direction | Pattern match similarity score |
| Compute cost | O(N × d) per query | O(N × m × log(m)) per pair |

### 4.2 Signal Blending Strategy

STUMPY produces a secondary signal that can be blended with the primary KNN signal:

```
KNN Signal: p_knn = 0.62 (BUY, confidence 0.62)
STUMPY Signal: p_stumpy = 0.58 (BUY, confidence 0.58)

Blended: p_combined = w_knn × p_knn + w_stumpy × p_stumpy
         where w_knn + w_stumpy = 1.0
         Default: w_knn = 0.80, w_stumpy = 0.20 (KNN is primary)

BSS gate: blended BSS ≥ KNN-only BSS + 0.003
```

### 4.3 Computational Budget

At 1500T, running AB-join for all ticker pairs is O(1500² × T × m) — prohibitive. Strategy:
1. Pre-filter: only run STUMPY on tickers where KNN signal fires (high confidence)
2. Restrict to same-sector candidates
3. Use `stumpy.stump()` (not all-pairs), one query series vs. one candidate at a time
4. Budget: ≤ 30 seconds per signal generation run (fits within 5-min execution window)

---

## 5. Autonomous EOD Architecture

### 5.1 Execution Timeline

```
     05:55  Task Scheduler launches eod_runner.bat
     06:00  ┌─ DATA PULL ──────────────── Yahoo Finance EOD prices ─────┐
     06:30  ├─ FEATURE REBUILD ─────────── prepare.py on updated data ──┤  OVERNIGHT
     07:00  ├─ INDEX REBUILD ──────────── HNSW/FAISS/TurboQuant build ──┤  BATCH
     08:00  └──────────────────────────────────────────────────────────── ┘
     09:00  ┌─ RECONCILIATION ─────────── Broker vs SharedState ────────┐
     09:30  ├─ HEALTH CHECK ───────────── Disk/memory/index freshness ──┤  PRE-MARKET
     10:00  └──────────────────────────────────────────────────────────── ┘
     ...    (Market hours — price streaming, stop-loss monitoring)
     15:55  ┌─ PRE-SIGNAL ────────────── Load index, warm caches ───────┐
     16:00  ├─ SIGNAL GENERATION ──────── Full pipeline → signals ──────┤  EOD
     16:15  ├─ ORDER SUBMISSION ───────── Submit via broker adapter ─────┤  EXECUTION
     16:30  ├─ CHECKPOINT ─────────────── SharedState save, daily report ┤
     16:45  └──────────────────────────────────────────────────────────── ┘
```

### 5.2 Failure Modes & Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Data pull timeout | Step timeout (300s) | Retry once; if still fails, use yesterday's data + YELLOW alert |
| Feature rebuild crash | Non-zero exit code | Halt pipeline; manual investigation required |
| Index build OOM | Process killed / TIMEOUT | Reduce batch size; rebuild with chunked processing |
| Reconciliation mismatch | > 0.05% drift | Block signal generation; manual review required |
| Health check FAIL | Non-zero exit code | Block signal generation; alert sent |
| Signal generation timeout | > 300s | Alert; skip today's signals |
| Broker API down | Connection error | Retry 3× with exponential backoff; alert if all fail |

### 5.3 Monitoring & Alerting

Phase 8 (paper trading) uses file-based logging. Phase 9 (live) adds:
- Webhook alerts (Slack/email) for FAIL/TIMEOUT steps
- Execution log analysis: `results/execution_log.jsonl`
- Weekly pipeline health report (as part of `scripts/weekly_review.py`)

---

## 6. Memory Budget at Scale

### 6.1 Projected Memory Usage

| Component | 585T | 1500T | 5200T | 5200T + 32D |
|-----------|------|-------|-------|-------------|
| Feature matrix (float64) | 116 MB | 296 MB | 1.6 GB | 6.4 GB |
| HNSW index (hnswlib) | ~200 MB | ~500 MB | ~3 GB | ~12 GB |
| FAISS IVF-Flat | — | ~340 MB | ~1.1 GB | ~4.4 GB |
| TurboQuant (3-bit) | 15 MB | 37 MB | 260 MB | 1.0 GB |
| Calibration arrays | 10 MB | 25 MB | 80 MB | 80 MB |
| Python overhead | ~500 MB | ~500 MB | ~500 MB | ~500 MB |
| **Total (HNSW)** | **~830 MB** | **~1.3 GB** | **~5.3 GB** | **~19 GB** |
| **Total (FAISS)** | — | **~1.2 GB** | **~3.5 GB** | **~11 GB** |
| **Total (FAISS+TQ)** | — | — | **~2.5 GB** | **~8 GB** |

**Conclusion:** At 5200T with 32D features, only FAISS + TurboQuant fits within the 24 GB budget. This validates the Phase 11 architecture decision.

### 6.2 Hard Budget: 24 GB Peak RAM

The 32 GB machine needs 8 GB headroom for OS + other processes. Index build is the peak memory moment (training data + index in memory simultaneously).

Mitigation strategies if budget is exceeded:
1. **Chunked index build:** Build FAISS IVF in chunks (train on subsample, add in batches)
2. **Memory-mapped features:** Use numpy memmap for feature matrix during index build
3. **Sector-partitioned indices:** One FAISS index per GICS sector (11 indices, each ~2.4M vectors)

---

## 7. Experiment Plan — TurboQuant Validation

### 7.1 Experiment Matrix

| Exp # | Dim | Bits | QJL | Metric | Gate |
|-------|-----|------|-----|--------|------|
| E1 | 8 | 1 | No | Recall@50 | Informational only |
| E2 | 8 | 2 | No | Recall@50 | ≥ 0.90 |
| E3 | 8 | 3 | No | Recall@50 | ≥ 0.95 |
| E4 | 8 | 4 | No | Recall@50 | ≥ 0.99 |
| E5 | 8 | 3 | Yes | Recall@50 | ≥ 0.99 (QJL should help) |
| E6 | 8 | 4 | Yes | Recall@50 | ≥ 0.999 |
| E7 | 8 | Best-from-above | — | BSS delta | ≥ -0.002 |
| E8 | 8 | Best | — | Walk-forward Sharpe | ≥ baseline - 0.05 |

### 7.2 Decision Tree

```
E1-E6 (recall sweep)
    │
    ├── Best recall ≥ 0.9990 at ≤ 4 bits?
    │   ├── YES → Run E7 (BSS delta)
    │   │         ├── BSS delta ≥ -0.002?
    │   │         │   ├── YES → Run E8 (Sharpe)
    │   │         │   │         ├── Sharpe holds? → Phase 6A PASSES → Proceed to 6B
    │   │         │   │         └── Sharpe drops > 0.05 → FAIL (cancel 6B)
    │   │         │   └── NO → FAIL (cancel 6B, revisit at higher d)
    │   │         └──
    │   └── NO → FAIL (cancel 6B, revisit at higher d)
    └──
```

### 7.3 Data for Experiments

Use the BSS-fixed baseline from Phase 1 (walk-forward fold data). This ensures:
1. We have exact BallTree neighbors as ground truth
2. We have BSS baseline to measure delta against
3. Results are directly comparable to production

---

## 8. Locked Settings — v2 Additions

| Parameter | Value | Source | Phase |
|-----------|-------|--------|-------|
| turbo_bits | 3 (default, adjustable by 6A results) | Phase 6A experiment | 6A |
| turbo_use_qjl | True | TurboQuant design | 6A |
| turbo_rotation_seed | 42 | Reproducibility | 6A |
| faiss_nprobe | 32 | FAISS heuristic for d=8 | 11 |
| faiss_n_cells | auto (sqrt(N)) | Standard FAISS | 11 |
| stumpy_subsequence_length | 20 | Standard matrix profile window | 7 |
| stumpy_weight | 0.20 | Signal blending (KNN primary) | 7 |
| eod_signal_timeout_s | 300 | 5-minute execution budget | 8+ |
| eod_hard_halt_timeout_s | 600 | 10-minute maximum | 8+ |

---

## Document Lineage

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-28 | Claude (Opus 4.6) | Initial TurboQuant/FAISS/STUMPY/EOD addendum |
