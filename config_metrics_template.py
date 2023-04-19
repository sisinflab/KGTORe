METRICS_TEMPLATE = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/train.tsv
    validation_path: ../data/{dataset}/val.tsv
    test_path: ../data/{dataset}/test.tsv
  dataset: {dataset}
  top_k: 10
  evaluation:
    cutoffs: [10]
    paired_ttest: True
    simple_metrics: [nDCGRendle2020, HR, Precision, Recall]
    # complex_metrics:
    #   - metric: clustered_nDCG
    #   user_clustering_name: WarmColdUsers
    #   user_clustering_file: ../data/{dataset}/users_deg.tsv
  gpu: 0
  models:
    RecommendationFolder:  
        folder: {recs}
"""
