import os.path

from data_preprocessing.filters.models import KGToreFilter, KaHFMFilter, KGINFilter, KGATFilter, KGFlexFilter
from data_preprocessing.filters.dataset import Binarize, Splitter
from data_preprocessing.filters import load_kg, load_dataset, load_linking, store_dataset, store_mapped_kg
from data_preprocessing.filters.knowledge import LinkingCleaning, KGTrainAlignment

dataset_relative_path = 'data/dataset.tsv'
kg_relative_path = 'dbpedia/triples.tsv'
linking_relative_path = 'dbpedia/linking.tsv'


def run(data_folder):
    print('\n***** facebook book data preparation *****\n'.upper())
    dataset_path = os.path.join(data_folder, dataset_relative_path)
    kg_path = os.path.join(data_folder, kg_relative_path)
    linking_path = os.path.join(data_folder, linking_relative_path)

    kg = load_kg(kg_path)
    dataset = load_dataset(dataset_path)
    linking = load_linking(linking_path)

    kwargs = {
        'kg': kg,
        'dataset': dataset,
        'linking': linking,
        'core': 5,
        'threshold': 0.97,
        'pred_kcore': 50
    }

    # data filtering
    binarizer = Binarize(dataset=dataset, threshold=1)
    kwargs['dataset'] = binarizer.filter()['dataset']

    # item-entity linking cleaning
    alignment = LinkingCleaning(linking=linking)
    kwargs.update(alignment.filter())

    paths = {}

    while True:

        flags = []

        kgtore = KGToreFilter(**kwargs)
        kwargs['dataset'] = kgtore.filter()['dataset']
        flags.append(kgtore.flag)
        paths['kgtore'] = store_mapped_kg(**kgtore._kwargs,
                                          folder=os.path.join(data_folder, 'kgtore'),
                                          name='kg',
                                          message='knowledge graph')
        del kgtore

        kgflex = KGFlexFilter(**kwargs)
        kwargs['dataset'] = kgflex.filter()['dataset']
        flags.append(kgflex.flag)
        paths['kgflex'] = store_mapped_kg(**kgflex._kwargs,
                                          folder=os.path.join(data_folder, 'kgflex'),
                                          name='kg',
                                          message='knowledge graph')
        store_dataset(kgflex._kwargs['item_features'],
                      folder=os.path.join(data_folder, 'kgflex'),
                      name='item_features',
                      message='item features')
        del kgflex

        kahfm = KaHFMFilter(**kwargs)
        kwargs['dataset'] = kahfm.filter()['dataset']
        flags.append(kahfm.flag)
        paths['kahfm'] = store_mapped_kg(**kahfm._kwargs,
                                         folder=os.path.join(data_folder, 'kahfm'),
                                         name='kg',
                                         message='knowledge graph')
        del kahfm

        kgin = KGINFilter(**kwargs)
        kwargs['dataset'] = kgin.filter()['dataset']
        flags.append(kgin.flag)
        paths['kgin'] = store_mapped_kg(**kgin._kwargs,
                                        folder=os.path.join(data_folder, 'kgin'),
                                        name='kg',
                                        message='knowledge graph')
        del kgin

        kgat = KGATFilter(**kwargs)
        kwargs['dataset'] = kgat.filter()['dataset']
        flags.append(kgat.flag)
        paths['kgat'] = store_mapped_kg(**kgat._kwargs,
                                        folder=os.path.join(data_folder, 'kgat'),
                                        name='kg',
                                        message='knowledge graph')
        del kgat

        if all(flags):
            break

    print(f'\nFinal transactions: {len(kwargs["dataset"])}')
    store_dataset(data=kwargs["dataset"],
                  folder=data_folder,
                  name='dataset',
                  message='dataset')

    print('\nThere will be the splitting...')

    splitter = Splitter(data=kwargs["dataset"],
                        test_ratio=0.2,
                        val_ratio=0.1)
    splitting_results = splitter.filter()
    print(f'Final training set transactions: {len(splitting_results["train"])}')
    print(f'Final test set transactions: {len(splitting_results["test"])}')
    print(f'Final validation set transactions: {len(splitting_results["val"])}')

    store_dataset(data=splitting_results["train"],
                  folder=data_folder,
                  name='train',
                  message='training set')

    store_dataset(data=splitting_results["test"],
                  folder=data_folder,
                  name='test',
                  message='test set')

    store_dataset(data=splitting_results["val"],
                  folder=data_folder,
                  name='val',
                  message='validation set')

    for model, model_paths in paths.items():
        model_kg = load_kg(model_paths['kg_path'], header=None)
        aligner = KGTrainAlignment(splitting_results['train'], model_kg)
        kg = aligner.filter()['kg']
        store_dataset(data=kg,
                      folder=os.path.join(data_folder, model),
                      name='kg',
                      message='knowledge graph filtered')
        del aligner
