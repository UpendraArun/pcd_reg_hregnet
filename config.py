import os 
import json

class DataConfig(object):
    def __init__(self, data) -> None:
        for k, v in data.items():
            setattr(self, k, v)

class ModelConfig(object):
    def __init__(self, data) -> None:
        for k, v in data.items():
            setattr(self, k, v)
            

class Config(object):
    def __init__(self, args) -> None:
        self.root = "/workspace/"
        self.dataset = args.dataset
        
        self.dataset_config = DataConfig(self.__read_json_file(os.path.join(self.root, 'dataset', 'config.json'))[self.dataset])
        #self.dataset_config.path = dataset_path
        self.dataset_config.epochs = args.epochs
        self.dataset_config.batch_size = args.batch_size
    
    def to_dict(self):
        config_dict = {key: value for key, value in self.__dict__.items() if not key.startswith("__")}

        if hasattr(self, 'dataset_config'):
            config_dict.update(self.dataset_config.__dict__)

        return config_dict

    @staticmethod
    def __read_json_file(file_path:str):
        if not os.path.exists(file_path):
            raise FileNotFoundError('Config file not found')

        with open(file_path, 'r') as file:
            data = json.load(file)
        return data 
