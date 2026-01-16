# Retail Analytics with Spark + HDFS

**Type II – Big Data query & analytics**

## Directory Layout

```bash
.
├─ cost/          # GCP cost estimate (CSV/screenshots)
├─ dataset/       # Local raw CSV: online_retail.csv (uploaded to HDFS)
├─ figures/       # Optional figures you pick for the report
├─ notebooks/     # Jupyter notebooks (all code)
├─ outputs/       # Generated PNG/CSV (mapped from /work/output)
├─ Proposal.docx
└─ stack.yml
```

## Stack

- Hadoop 3.2.x (HDFS on `namenode:9000`)
- Spark 3.0.x (Standalone `spark-master:7077`)
- Jupyter Notebook (PySpark)
- Data lake style: raw CSV → clean → Parquet partitions; compute on Spark, results to `outputs/`

## Quick Start

```bash
# 1) Bring up the cluster
docker compose -f stack.yml -p infs3208 up -d

# 2) Upload CSV to HDFS
docker compose -f stack.yml -p infs3208 exec namenode hdfs dfs -mkdir -p /datasets
docker compose -f stack.yml -p infs3208 exec namenode \
  hdfs dfs -put -f /host_dataset/online_retail.csv /datasets/

# 3) Open Jupyter (http://localhost:8888 or check token in logs)
```

> Paths used in code:
>
> - `CSV_PATH = "hdfs://namenode:9000/datasets/online_retail.csv"`
> - `PARQ_PATH = "hdfs://namenode:9000/datasets/retail_clean_parquet"`
> - `LOCAL_OUT = "/work/output"` (mapped to `outputs/`)

## Execution Flow (in `notebooks/`)

1. **Clean & Parquet** (filter cancellations/anomalies; normalize timestamps).
2. **EDA & Charts** (Top countries/products; monthly trend with dual axis).
3. **Basket Co-occurrence** (pair count / support / confidence / lift; export Top-N).
4. **RFM + KMeans** (standardize → k=4; report train/test **silhouette** & cluster stats).
5. **ALS Recs + Eval** (implicit feedback; Top-K; **Precision/Recall/HitRate/NDCG/MRR**).
6. **ABC / Pareto** (cumulative revenue curve & A/B/C bar chart).

## Key Artifacts (all in `outputs/`)

- `*_top10_*.csv`, `monthly_trend.csv`
- `als_eval_metrics.csv`
- Plots:
  - `plot_top10_countries_count*.png`, `plot_monthly_trend.png`
  - `plot_pairs_top_by_count.png`, `pairs_top20_by_*.csv`
  - `plot_rfm_f_vs_m*.png`, `plot_rfm_cluster_counts.png`
  - `plot_product_abc_counts.png`, `plot_product_pareto.png`
  - `plot_als_most_recommended_full.png`

## Cost & Resource Assumptions

- See `cost/` (GCP Pricing Calculator export):
  - master: e2-standard-2, **50 GiB** boot disk
  - two workers: e2-standard-2 (preemptible allowed), **50 GiB** boot disk each
- Region: `australia-southeast1 (Sydney)`; totals documented in CSV/screenshots.

## Reproducibility Knobs

- `SEED=42`, ALS `K=10`, `TOPK_RECS=100`.
- Spark session enforces:
  - `spark.pyspark.python=python3`
  - `spark.pyspark.driver.python=python3`
    (to prevent driver/executor Python mismatch)