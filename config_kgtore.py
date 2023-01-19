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
      batch_size: 64
      lr: [ loguniform, -9.210340372, -5.298317367]
      elr: [ loguniform, -9.210340372, -5.298317367 ]
      alpha: {alpha}
      beta: {beta}
      gamma: [ uniform, 0, 1]
      epochs: 200
      factors: 64
      l_w: [ loguniform, -11.512925465, -2.30258509299 ]
      n_layers: {layers}
      npr: {npr}
      criterion: {strategy}
      loader: KGTORETSVLoader
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: nDCGRendle2020@10
        verbose: True
"""
