from datasets import load_dataset, Dataset as HFDataset 
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import DataConfig
import numpy as np
import os

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
    def __init__(
        self,
        config: DataConfig,
        ts_dataset: TinyStories,
        split: str ="train",
        block_size: int = 1024,
        filename_base: str = "tiny_stories"
    ):
        
        assert split in ["train", "validation"], f"Error creating the dataset. Split should be either train or validation. Got {split}."
        
        self.config = config
        self.block_size = block_size
        self.split = split
        self.filename_base = filename_base
        self.data = None
        
        if os.path.exists(f'{self.filename_base}_{split}.npy'):
            print(f"Loading {split} dataset from disk...")
            self.data = np.load(f'{self.filename_base}_{split}.npy', mmap_mode='r')
            print("Dataset loaded")
        else:
            if split == "train":
                self.data = ts_dataset.get_train()
            elif split == "validation":
                self.data = ts_dataset.get_validation()
            
            self.encode()
            
    def __len__(self) -> int:
        return len(self.data) // self.block_size
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        # get a block of bytes
        start = idx * self.block_size
        end = start + self.block_size
        bytes_ids = self.data[start:end]
        
        # span corruption
        source_ids, target_ids = self.span_corruption(bytes_ids)
        
        # add EOS
        source_ids.append(self.config.eos_token_id)
        target_ids.append(self.config.eos_token_id)
        
        # convert to tensor
        source = torch.tensor(source_ids, dtype=torch.long)
        target = torch.tensor(target_ids, dtype=torch.long)
        
        return source, target
    
    def encode(self):
        # flatten
        stories = [i['text'] for i in self.data]
        
        all_bytes = []
        for story in stories:
            # Encode story and append eos token
            s = list(story.encode("utf-8"))
            s.append(self.config.eos_token_id)
            all_bytes.extend(s)
        
        # save to disk
        print(f"Saving {self.split} dataset to disk...")
        np.save(f'{self.filename_base}_{self.split}.npy', np.array(all_bytes, dtype=np.uint16))
        print(f"Dataset saved in {self.filename_base}.npy")
        self.data = np.load(f'tiny_stories_{self.split}.npy', mmap_mode='r')
        
    def span_corruption(self, x: np.ndarray) -> tuple[list]:
        
        # track masked bytes
        is_masked = np.zeros(len(x), dtype=bool)
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
            while is_masked[start_idx : start_idx + span_length].any():
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
        masked_spans.sort(key=lambda span:span[0])
        
        source_ids = []
        target_ids = []
        
        current_pos = 0
        for span in masked_spans:
            target_ids.append(span[2]) # sentinel
            target_ids.extend(x[span[0]:span[1]]) # corrupted bytes
            
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
        
        # create decoder input by prepending PAD and removing last token
        decoder_inputs = []
        for t in targets:
            pad_tensor = torch.tensor([self.pad_token_id], dtype=t.dtype)
            dec_input = torch.cat([pad_tensor, t[:-1]])
            decoder_inputs.append(dec_input)
        
        # pad sequences
        padded_sources = pad_sequence(
            sources,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        
        padded_decoder_inputs = pad_sequence(
            decoder_inputs,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        
        padded_targets = pad_sequence(
            targets,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        
        return padded_sources, padded_decoder_inputs, padded_targets
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    config = DataConfig()
    ds = TinyStoriesDataset(config, ts_dataset=TinyStories(), split="train")
    collator = PadCollator(config.pad_token_id)
    
    loader = DataLoader(ds, batch_size=2, collate_fn=collator)
    
    for x, dec_input, y in loader:
        print(x)
        print("------")
        print(dec_input)
        print("------")
        print(y)
        break