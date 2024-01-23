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
  dataset: facebook_book
  top_k: 10
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.KGTORE:
      meta:
        verbose: True
        save_recs: False
        validation_rate: 1
        validation_metric: nDCGRendle2020@10
      batch_size: 64
      lr: 0.0007327526296111888
      elr: 0.0029515807011013507
      l_w: 0.020755811015626745
      alpha: {alpha}
      depth: {depth}
      epochs: 100
      factors: 64
      n_layers: 3
      npr: {npr}
      loader: KGTORETSVLoader
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: nDCGRendle2020@10
        verbose: True
        '''