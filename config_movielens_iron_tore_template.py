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
  dataset: movielens
  top_k: 10
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.KGTORE:
      meta:
        hyper_max_evals: 20
        hyper_opt_alg: tpe
        verbose: True
        save_recs: False
        validation_rate: 1
        validation_metric: nDCGRendle2020@10
      batch_size: 2048
      lr: [ loguniform, -9.721165996, -5.11599581 ]
      elr: [ loguniform, -8.111728083, -3.506557897 ]
      l_w: [ loguniform, -8.111728083, -3.506557897 ]
      alpha: {alpha}
      epochs: 100
      factors: 64
      n_layers: 3
      npr: 1
      loader: KGTORETSVLoader
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: nDCGRendle2020@10
        verbose: True'''