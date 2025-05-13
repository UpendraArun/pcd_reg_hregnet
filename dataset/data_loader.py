from .man_dataset import TruckScenesDataset, TruckScenesLoader, TruckScenesPerturbation
from .audi_dataset import A2D2Dataset, A2D2Perturbation
from config import Config

def load_dataset(config, split):
    if config.dataset == 'man':
        config.dataset_config.split = split
        
        if config.dataset_config.split == 'test':
            config.dataset_config.version = 'v1.0-test'
            config.dataset_config.path = "/workspace/data/truckscenes_test"
            config.dataset_config.perturbations_file = "/workspace/data/truckscenes_test/perturbations_file_"
        
        loader = TruckScenesLoader()
        man_data = TruckScenesDataset(loader(config, verbose=False), config)
        man_data_perturb = TruckScenesPerturbation(dataset=man_data, config=config)

        return man_data_perturb 
    
    elif config.dataset == 'audi':
        config = Config(dataset='audi')
        config.dataset_config.split = split
        a2d2_data = A2D2Dataset(config=config)
        a2d2_data_perturb = A2D2Perturbation(dataset=a2d2_data, config=config)
        return a2d2_data_perturb
    
    if config.dataset == 'kitti':
        train_seqs = ['00','01','02','03','04','05']
        #train_dataset = KittiDataset(args.root, train_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    elif config.dataset == 'nuscenes':
        train_seqs = ['train']
        #train_dataset = NuscenesDataset(args.root, train_seqs, args.npoints, args.voxel_size, args.data_list, args.augment)
    
    else:
        raise NotImplementedError(f"Dataset {config.dataset} not implemented")
