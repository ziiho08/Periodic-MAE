""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data.distributed
import torch

RANDOM_SEED = 64
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)

train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/intra/1PURE_RHYTHMFORMER.yaml", type=str, help="The name of the model.")
    '''Neural Method Sample YAML LIST:
      SCAMPS_SCAMPS_UBFC-rPPG_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_PHYSNET_BASIC.yaml
      SCAMPS_SCAMPS_PURE_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_PURE_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_PURE_PHYSNET_BASIC.yaml
      PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml
      PURE_PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      PURE_PURE_UBFC-rPPG_PHYSNET_BASIC.yaml
      PURE_PURE_MMPD_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_PHYSNET_BASIC.yaml
      MMPD_MMPD_UBFC-rPPG_TSCAN_BASIC.yaml
    Unsupervised Method Sample YAML LIST:
      PURE_UNSUPERVISED.yaml
      UBFC-rPPG_UNSUPERVISED.yaml
    '''
    return parser


def train_and_test(config, data_loader_dict):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'RhythmFormer':
        model_trainer = trainer.RhythmFormerTrainer.RhythmFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PeriodicMAE':
        model_trainer = trainer.PeriodicMAETrainer.PeriodicMAETrainer(config, data_loader_dict)    
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    """Tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict)    
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'RhythmFormer':
        model_trainer = trainer.RhythmFormerTrainer.RhythmFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PeriodicMAE':
        model_trainer = trainer.PeriodicMAETrainer.PeriodicMAETrainer(config, data_loader_dict)    
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader_dict)


def unsupervised_method_inference(config, data_loader):
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        if unsupervised_method == "POS":
            unsupervised_predict(config, data_loader, "POS")
        elif unsupervised_method == "CHROM":
            unsupervised_predict(config, data_loader, "CHROM")
        elif unsupervised_method == "ICA":
            unsupervised_predict(config, data_loader, "ICA")
        elif unsupervised_method == "GREEN":
            unsupervised_predict(config, data_loader, "GREEN")
        elif unsupervised_method == "LGI":
            unsupervised_predict(config, data_loader, "LGI")
        elif unsupervised_method == "PBV":
            unsupervised_predict(config, data_loader, "PBV")
        elif unsupervised_method == "OMIT":
            unsupervised_predict(config, data_loader, "OMIT")
        else:
            raise ValueError("Not supported unsupervised method!")


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    data_loader_dict = dict() # dictionary of data loaders 
    if config.TOOLBOX_MODE == "train_and_test":
        # train_loader
        if config.TRAIN.DATA.DATASET == "UBFCrPPG":
            train_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.TRAIN.DATA.DATASET == "PURE":
            train_loader = data_loader.PURELoader.PURELoader
        elif config.TRAIN.DATA.DATASET == "SCAMPS":
            train_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TRAIN.DATA.DATASET == "MMPD":
            train_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TRAIN.DATA.DATASET == "BP4DPlus":
            train_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.TRAIN.DATA.DATASET == "BP4DPlusBigSmall":
            train_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.TRAIN.DATA.DATASET == "UBFC-PHYS":
            train_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.TRAIN.DATA.DATASET == "iBVP":
            train_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.TRAIN.DATA.DATASET == "V4V":
            train_loader = data_loader.V4VLoader.V4VLoader            
        elif config.TRAIN.DATA.DATASET == "Periodic_MAE":
            data1 = data_loader.PURELoader.PURELoader(
                name="train", 
                data_path=config.TRAIN.DATA.PURE.DATA_PATH, 
                config_data=config.TRAIN.DATA.PURE
            )
            data2 = data_loader.MMPDLoader.MMPDLoader(
                name="train", 
                data_path=config.TRAIN.DATA.MMPD.DATA_PATH, 
                config_data=config.TRAIN.DATA.MMPD
            )
            data3 = data_loader.UBFCrPPGLoader.UBFCrPPGLoader(
                name="train", 
                data_path=config.TRAIN.DATA.UBFCrPPG.DATA_PATH, 
                config_data=config.TRAIN.DATA.UBFCrPPG
            )
            merged_dataset = ConcatDataset([data1, data2, data3])
            print("===Dataset Merged!===")
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFCrPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP.")

        if config.TRAIN.DATA.DATASET == "Periodic_MAE":
            data_loader_dict['train'] = DataLoader(
                dataset=merged_dataset,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=train_generator
            )
        else:
            train_data_loader = train_loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA)
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=train_generator
            )

        # valid_loader
        if config.VALID.DATA.DATASET == "UBFCrPPG":
            valid_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.VALID.DATA.DATASET == "PURE":
            valid_loader = data_loader.PURELoader.PURELoader
        elif config.VALID.DATA.DATASET == "SCAMPS":
            valid_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.VALID.DATA.DATASET == "MMPD":
            valid_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.VALID.DATA.DATASET == "BP4DPlus":
            valid_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.VALID.DATA.DATASET == "BP4DPlusBigSmall":
            valid_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.VALID.DATA.DATASET == "UBFC-PHYS":
            valid_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.VALID.DATA.DATASET == "iBVP":
            valid_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.VALID.DATA.DATASET == "V4V":
            valid_loader = data_loader.V4VLoader.V4VLoader
        elif config.VALID.DATA.DATASET is None and not config.TEST.USE_LAST_EPOCH:
            raise ValueError("Validation dataset not specified despite USE_LAST_EPOCH set to False!")
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFCrPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP")
        
        if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH):
            valid_data = valid_loader(
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['valid'] = None

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
        # test_loader
        if config.TEST.DATA.DATASET == "UBFCrPPG":
            test_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.TEST.DATA.DATASET == "PURE":
            test_loader = data_loader.PURELoader.PURELoader
        elif config.TEST.DATA.DATASET == "SCAMPS":
            test_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TEST.DATA.DATASET == "MMPD":
            test_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TEST.DATA.DATASET == "BP4DPlus":
            test_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.TEST.DATA.DATASET == "BP4DPlusBigSmall":
            test_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.TEST.DATA.DATASET == "UBFC-PHYS":
            test_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.TEST.DATA.DATASET == "iBVP":
            test_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.TEST.DATA.DATASET == "V4V":
            test_loader = data_loader.V4VLoader.V4VLoader     
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFCrPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP.")
        
        if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
            print("Testing uses last epoch, validation dataset is not required.", end='\n\n')   

        if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=16,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['test'] = None

    elif config.TOOLBOX_MODE == "unsupervised_method":
        # unsupervised method dataloader
        if config.UNSUPERVISED.DATA.DATASET == "UBFCrPPG":
            unsupervised_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.UNSUPERVISED.DATA.DATASET == "PURE":
            unsupervised_loader = data_loader.PURELoader.PURELoader
        elif config.UNSUPERVISED.DATA.DATASET == "SCAMPS":
            unsupervised_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "MMPD":
            unsupervised_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.UNSUPERVISED.DATA.DATASET == "BP4DPlus":
            unsupervised_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.UNSUPERVISED.DATA.DATASET == "UBFC-PHYS":
            unsupervised_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "iBVP":
            unsupervised_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.UNSUPERVISED.DATA.DATASET == "V4V":
            unsupervised_loader = data_loader.V4VLoader.V4VLoader     
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFCrPPG, PURE, MMPD, \
                             SCAMPS, BP4D+, UBFC-PHYS and iBVP.")
        
        unsupervised_data = unsupervised_loader(
            name="unsupervised",
            data_path=config.UNSUPERVISED.DATA.DATA_PATH,
            config_data=config.UNSUPERVISED.DATA)
        data_loader_dict["unsupervised"] = DataLoader(
            dataset=unsupervised_data,
            num_workers=16,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=general_generator
        )

    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test or only_test or unsupervised_method.")

    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "unsupervised_method":
        unsupervised_method_inference(config, data_loader_dict)
    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !", end='\n\n')