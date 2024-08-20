import os
import argparse
from copy import deepcopy
import pickle
from pathlib import Path
from argparse import Namespace

import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from models import DrFuseTrainer
from utils import EHRDiscretizer, EHRNormalizer, get_ehr_datasets, load_cxr_ehr, load_discretized_header


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('--data_dir', type=str, default='./dataset/mimiciv_ehr_cxr')
    parser.add_argument('--cxr_resized_data_dir', type=str, default='/hdd2/mimic_cxr_resized')
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--data_pair', type=str, default='paired', choices=['partial', 'paired'])
    parser.add_argument('--timestep', type=float, default=1.0)
    parser.add_argument('--lambda_disentangle_shared', type=float, default=1)
    parser.add_argument('--lambda_disentangle_ehr', type=float, default=1)
    parser.add_argument('--lambda_disentangle_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_ehr', type=float, default=1)
    parser.add_argument('--lambda_pred_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_shared', type=float, default=1)
    parser.add_argument('--aug_missing_ratio', type=float, default=0.3)
    parser.add_argument('--lambda_attn_aux', type=float, default=1)
    parser.add_argument('--ehr_n_layers', type=int, default=1)
    parser.add_argument('--ehr_n_head', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--normalizer_state', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()


    # set number of threads allowed
    torch.set_num_threads(5)

    # set seed
    L.seed_everything(args.seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_root = Path(args.data_dir)
    ehr_dir = data_root / 'ehr'
    cxr_dir = data_root / 'cxr'

    discretizer = EHRDiscretizer(timestep=1.0,
                                 store_masks=True,
                                 impute_strategy='previous',
                                 start_time='zero')

    # all non-categorical variables

    discretizer_header = load_discretized_header(discretizer=discretizer, ehr_dir=ehr_dir,
                                                 txt_fpath=ehr_dir/'discretizer_header.txt')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = EHRNormalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = 'normalizers/ph_ts{}.input_str_previous.start_time_zero.normalizer'.format(args.timestep)
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)

    # normalizer = None

    ehr_datasets = get_ehr_datasets(discretizer, normalizer,
                                    ehr_data_dir=ehr_dir,
                                    ehr_pkl_fpath_train=ehr_dir/'ehr_phenotyping_48h_train.pkl',
                                    ehr_pkl_fpath_val=ehr_dir/'ehr_phenotyping_48h_val.pkl',
                                    ehr_pkl_fpath_test=ehr_dir/'ehr_phenotyping_48h_test.pkl')

    dataloaders = load_cxr_ehr(cxr_resized_data_dir=args.cxr_resized_data_dir,
                               data_pairs_train=args.data_pair,
                               data_ratio=args.data_ratio,
                               batch_size=args.batch_size,
                               meta_pkl_fpath=data_root/'metas_with_labels_phenotyping_first_48h.pkl',
                               ehr_datasets=ehr_datasets,
                               cxr_pkl_fpath_train=cxr_dir/'cxr_phenotyping_48h_train.pkl',
                               cxr_pkl_fpath_val=cxr_dir/'cxr_phenotyping_48h_val.pkl',
                               cxr_pkl_fpath_test=cxr_dir/'cxr_phenotyping_48h_test.pkl',
                               num_workers=args.num_workers)

    train_dl, val_dl, test_dl_partial, test_dl_paired = dataloaders

    model = DrFuseTrainer(args=args, label_names=train_dl.dataset.CLASSES)

    callback_metric = 'val_PRAUC_avg_over_dxs/final'
    early_stop_callback = EarlyStopping(monitor=callback_metric,
                                        min_delta=0.00,
                                        patience=args.patience,
                                        verbose=False,
                                        mode="max")


    trainer = L.Trainer(devices=[0],
                        accelerator='gpu',
                        max_epochs=args.epochs,
                        min_epochs=min(args.epochs, 10),
                        callbacks=[early_stop_callback])

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # do testing
    results = {
        'best_val_prauc': trainer.callback_metrics['val_PRAUC_avg_over_dxs/final'].item(),
        'best_val_roauc': trainer.callback_metrics['val_AUROC_avg_over_dxs/final'].item()
    }
    logpath = trainer.logger.log_dir
    trainer.loggers = None
    trainer.test(model=model, dataloaders=test_dl_partial)
    results['partial_test_results'] = deepcopy(model.test_results)

    trainer.test(model=model, dataloaders=test_dl_paired)
    results['paired_test_results'] = deepcopy(model.test_results)

    with open(logpath+'/test_results', 'wb') as f:
        pickle.dump(results, f)
