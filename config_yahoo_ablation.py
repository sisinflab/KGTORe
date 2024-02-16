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
    simple_metrics: [nDCGRendle2020, HR, Precision, Recall]
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
      batch_size: 256
      lr: [ loguniform, -9.2103403719, -5.298317 ]
      elr: [ loguniform, -6,907755278982137, -3.506557897319982 ]
      l_w: [ loguniform, -7.600902459542082, -3.912023005428146 ]
      epochs: 200
      factors: 64
      l_ind: 0
      alpha: {alpha}
      beta: {beta}
      ind_edges: 0
      n_layers: 3
      npr: 20
      aggr: {aggr}
      mode: abl_rnd
      loader: KGTORETSVLoader
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: nDCGRendle2020@10
        verbose: True
'''