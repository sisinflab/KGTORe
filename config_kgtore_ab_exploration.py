KGTORE_CONFIG = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{dataset}/train.tsv
    validation_path: ../data/{dataset}/val.tsv
    test_path: ../data/{dataset}/test.tsv
    side_information:
      - dataloader: KGTORETSVLoader
        kg: ../data/{dataset}/kgtore/kg.tsv
  dataset: {dataset}
  top_k: 10
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020]
  gpu: {gpu}
  external_models_path: ../external/models/__init__.py
  models:
    external.KGTORE:
      meta:
        verbose: True
        save_recs: False
        validation_rate: 1
        validation_metric: nDCGRendle2020@10
      batch_size: {batch}
      lr: 0.000664750387531533
      elr: 0.00305727041704464
      l_w: 0.00312734853526998
      alpha: {alpha}
      beta: {beta}
      gamma: {gamma}
      ind_edges: {ind_edges}
      epochs: 200
      factors: 64
      n_layers: 3
      npr: 20
      criterion: entropy
      loader: KGTORETSVLoader
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: nDCGRendle2020@10
        verbose: True
"""
