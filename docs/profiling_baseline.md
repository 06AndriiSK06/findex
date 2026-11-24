# Baseline Profiling Report (Before NumPy Optimization)

- **Flame graphs:** `docs/profile_index.svg`, `docs/profile_search.svg`
- **Index workload:** `build_index("data/lab06_bench/crawl_c20.jsonl", workers=1)` ? 200 docs, 15,147 terms (`0.706 s`)
- **Search workload:** 300 uncached runs of `"python async function"` (`k=10`) on `index.bin` ? 1,015 docs, 55,764 terms (`0.410 s` total / `1.37 ms` per query)

## Top 3 Hotspots & Hypotheses (Recorded Before Code Changes)

| Rank | Function & Location | Cumulative Time | Execution Breakdown | Hypothesis & Optimization Plan |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `index.py:110` (`intersect_merge`) | `0.178 s` (43.4% of search) | ~90% Python / ~10% Native (`len`) | Two-pointer `while` loop over Python `Posting` lists executes >1.1M `len()` calls and `Posting.doc_id` attribute lookups. Replace with vectorized `np.intersect1d` over contiguous `int32` `doc_ids` arrays. |
| **2** | `ranking.py:34` (`BM25.score`) | `0.131 s` (32.0% of search, 72,000 calls) | ~88% Python / ~12% Native (`math.log`) | Scalar Python loop recomputes `math.log`, looks up `index.df()` and `index.doc_length()` 72,000 times, and boxes `float` objects. Replace with a single vectorized NumPy expression (`ufuncs` + fancy indexing `doc_lengths[doc_ids]`) and `np.argpartition` top-k. |
| **3** | `index.py:152` (`_build_partial_from_docs`) & `tokenize.py:10` (`tokenize`) | `0.638 s` (90.4% of build_index) | ~65% Python / ~35% Native (`re.Match`, `casefold`) | Per-token generator yields and `Posting` object allocations across 520,684 tokens dominate indexing time. Convert posting lists to contiguous `np.ndarray` (`int32`) immediately after indexing/deserialization. |
