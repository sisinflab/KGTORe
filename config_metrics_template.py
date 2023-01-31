METRICS_TEMPLATE = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/train.tsv
    validation_path: ../data/{dataset}/val.tsv
    test_path: ../data/{dataset}/test.tsv
  dataset: {dataset}
  top_k: {k}
  evaluation:
    cutoffs: [{k}]
    # paired_ttest: True
    # wilcoxon_test: True
    simple_metrics: [nDCGRendle2020, nDCG, HR, Precision, Recall, MAP, MRR, ItemCoverage, UserCoverage, NumRetrieved,
                     UserCoverage, Gini, SEntropy, EFD, EPC, PopREO, PopRSP, ACLT, APLT, ARP]
  gpu: 0
  models:
    RecommendationFolder:  
        folder: {recs}
"""
