from datasets import load_dataset, Dataset as HFDataset 
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import DataConfig
import numpy as np
from copy import deepcopy

class TinyStories:
    def __init__(self, dataset_id: str = "roneneldan/TinyStories"):
        self.dataset_id = dataset_id
        self.dataset = load_dataset(self.dataset_id)
        
    def get_train(self) -> HFDataset:
        if 'train' in self.dataset:
            return self.dataset['train']
        
        raise ValueError("Train split not found in the dataset")
    
    def get_validation(self) -> HFDataset:
        if 'validation' in self.dataset:
            return self.dataset['validation']
        
        raise ValueError("Validation split not found in the dataset")
    
class TinyStoriesDataset(Dataset):
    def __init__(self, config: DataConfig, ts_dataset: TinyStories, split: str ="train"):
        
        assert split in ["train", "validation"], f"Error creating the dataset. Split should be either train or validation. Got {split}."
        
        self.config = config
        self.data = None
        if split == "train":
            self.data = ts_dataset.get_train()
        elif split == "validation":
            self.data = ts_dataset.get_validation()
            
            
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        story = self.data[idx]['text']
        
        # convert to a list of utf-8 bytes
        bytes_ids = list(story.encode("utf-8"))
        
        # span corruption
        source_ids, target_ids = self.span_corruption(bytes_ids)
        
        # add EOS
        source_ids.append(self.config.eos_token_id)
        target_ids.append(self.config.eos_token_id)
        
        # convert to tensor
        source = torch.tensor(source_ids, dtype=torch.long)
        target = torch.tensor(target_ids, dtype=torch.long)
        
        return source, target
    
    def span_corruption(self, x: list) -> tuple[list]:
        
        # track masked bytes
        is_masked = np.zeros((1, len(x)), dtype=bool)
        masked_spans = [] # contains span as a tuple (start_idx, end_idx, sentinel_id)
        
        # number of token to mask
        total_to_mask = round(len(x) * self.config.mask_pct)
        # token already masked
        masked_count = 0
        # sentinel id. Starts from 255 and goes down
        sentinel_id = 255
        
        while masked_count < total_to_mask:
            # Generate span length using poisson distribution
            span_length = np.random.poisson(lam=self.config.mean_span_corruption_length)
            span_length = max(span_length, 1)
            
            # choose start index
            start_idx = np.random.randint(0, len(x) - span_length)
            # prevent overlapping spans
            while any(is_masked[start_idx : start_idx + span_length]):
                start_idx = np.random.randint(0, len(x) - span_length)
            
            is_masked[start_idx:start_idx + span_length] = True
                
            masked_spans.append((
                start_idx,
                start_idx + span_length,
                sentinel_id
            ))
            
            masked_count += span_length
            sentinel_id -= 1
        
        # sort by start index
        masked_spans.sort(key=lambda span: span[0])
        
        source_ids = []
        target_ids = []
        
        current_pos = 0
        for span in masked_spans:
            target_ids.append(span[2]) # sentinel
            target_ids.extend(x[span[0]: span[1]]) # corrupted bytes
            
            source_ids.extend(x[current_pos:span[0]]) # non masked bytes
            source_ids.append(span[2]) # sentinel
            current_pos = span[1] # move position to the end of span
            
        source_ids.extend(x[current_pos:])
            
        return source_ids, target_ids
        

class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
    
        # separate source and target
        sources = [i[0] for i in batch]
        targets = [i[1] for i in batch]
        
        # pad sequences
        padded_sources = pad_sequence(
            sources,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        
        padded_targets = pad_sequence(
            targets,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        
        return padded_sources, padded_targets
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    config = DataConfig()
    ds = TinyStoriesDataset(config, ts_dataset=TinyStories(), split="train")
    collator = PadCollator(config.pad_token_id)
    
    loader = DataLoader(ds, batch_size=2, collate_fn=collator)
    
    for x, y in loader:
        print(x)
        print("------")
        print(y)
        break