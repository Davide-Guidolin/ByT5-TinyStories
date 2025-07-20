from datasets import load_dataset, Dataset 

class Data:
    def __init__(self, dataset_id: str = "roneneldan/TinyStories"):
        self.dataset_id = dataset_id
        self.dataset = load_dataset(self.dataset_id)
        
    def get_train(self) -> Dataset:
        if 'train' in self.dataset:
            return self.dataset['train']
        
        raise ValueError("Train split not found in the dataset")
    
    def get_validation(self) -> Dataset:
        if 'validation' in self.dataset:
            return self.dataset['validation']
        
        raise ValueError("Validation split not found in the dataset")    
    
if __name__ == "__main__":
    data = Data()
    print(data.get_train()[0])
    print(data.get_validation()[0])

