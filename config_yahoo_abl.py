TEMPLATE = '''experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../data/{0}/train.tsv
    validation_path: ../data/{0}/val.tsv
    test_path: ../data/{0}/test.tsv
    side_information:
      - dataloader: KGTORETSVLoader
        kg: ../data/{0}/kgtore/kg.tsv
  dataset: yahoo_movies
  top_k: 10
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020, HR, Precision, Recall, ItemCoverage, SEntropy, Gini]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.KGTORE:
      meta:
        verbose: True
        save_recs: False
        validation_rate: 1
        validation_metric: nDCGRendle2020@10
      batch_size: 256
      lr: 0.00016283304013652814
      elr: 0.001679893762698589
      l_w: 0.019122503451020175
      alpha: {alpha}
      npr: {npr}
      abl: {abl}
      epochs: 100
      factors: 64
      n_layers: 3
      loader: KGTORETSVLoader
      seed: {seed}
      early_stopping:
        patience: 5
        mode: auto
        monitor: nDCGRendle2020@10
        verbose: True'''