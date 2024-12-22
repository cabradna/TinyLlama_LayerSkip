import json
import re
import json
import torch
from torch.utils.data import random_split
import os
from tqdm import tqdm
import pickle

class ConversationalCodeDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, max_length=1024):
        """
        Dataset for conversational code data with code extraction capability
        
        Args:
            json_file (str): Path to JSON file containing conversations
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Process conversations and code
        self.processed_data = []
        for item in self.data:
            # Format and process the conversation
            processed_item = self.process_conversation(item)
            self.processed_data.append(processed_item)

    def process_conversation(self, item):
        """
        Process conversation and extract code blocks
        Returns processed item with masked labels
        
        Args:
            item (dict): Dictionary containing 'instruction' and 'output' fields
        """
        # Format question and answer separately
        question = f"Question: {item['instruction']}"
        answer = f"Answer: {item['output']}"
        formatted = f"{question}\n{answer}"
        
        # Get question length in tokens
        question_tokens = self.tokenizer(
            question,
            add_special_tokens=True,  # Include any special tokens that would be added
            return_tensors='pt'
        )
        question_length = question_tokens['input_ids'].size(1)
        
        # Tokenize full conversation
        encodings = self.tokenizer(
            formatted,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels and mask question portion
        labels = encodings['input_ids'].squeeze(0).clone()
        labels[:question_length] = -100  # Standard ignore_index for CrossEntropyLoss
        
        code_blocks = self.extract_code_blocks(formatted)
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': labels,
            'original_text': formatted,
            'code_blocks': code_blocks,
            'segments': self.extract_segments(formatted)
        }

    def extract_code_blocks(self, text):
        """Extract code blocks from text"""
        # Match code blocks between triple backticks
        code_blocks = re.findall(r'```(?:python)?(.*?)```', text, re.DOTALL)
        return [block.strip() for block in code_blocks]

    def extract_segments(self, text):
        segments = []
        parts = text.split('\n\n')
        
        for part in parts:
            if part:
                if part.startswith('```') and part.endswith('```'):
                    # Add a check for the regex match
                    match = re.match(r'```(?:python)?(.*?)```', part, re.DOTALL)
                    if match:  # Only proceed if there's a match
                        code = match.group(1).strip()
                        segments.append(('code', code))
                    else:
                        # Handle cases where the code block format doesn't match exactly
                        code = part.strip('`').strip()
                        segments.append(('code', code))
                else:
                    segments.append(('text', part.strip()))
        
        return segments

    def get_code_context(self, idx):
        """
        Get code blocks with their surrounding context
        Useful for evaluation
        """
        item = self.processed_data[idx]
        return {
            'full_text': item['original_text'],
            'code_blocks': item['code_blocks'],
            'segments': item['segments']
        }

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Returns the tokenized data for training
        but maintains access to extracted code through get_code_context
        """
        item = self.processed_data[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': item['labels']
        }

class DatasetPreprocessor:
    def __init__(self, json_file, tokenizer, max_length=1024, 
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                 output_dir="processed_datasets"):
        """
        Preprocesses and splits the dataset
        
        Args:
            json_file (str): Path to JSON file
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length
            train_ratio (float): Proportion of training data
            val_ratio (float): Proportion of validation data
            test_ratio (float): Proportion of test data
            output_dir (str): Directory to save processed datasets
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        
        self.json_file = json_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.output_dir = output_dir
        self.splits = {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def process_and_save_datasets(self):
        """Process the full dataset and save train/val/test splits"""
        print("Loading and processing dataset...")
        
        # Create full dataset
        full_dataset = ConversationalCodeDataset(
            json_file=self.json_file,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(self.splits['train'] * total_size)
        val_size = int(self.splits['val'] * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Save each split
        splits = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        for split_name, dataset in splits.items():
            self._save_split(dataset, split_name)
            
        # Save split information
        split_info = {
            'total_size': total_size,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'max_length': self.max_length
        }
        
        with open(os.path.join(self.output_dir, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
            
        print("Dataset processing complete!")
        print(f"Train size: {train_size}")
        print(f"Validation size: {val_size}")
        print(f"Test size: {test_size}")

    def _save_split(self, dataset, split_name):
        """Save a dataset split to disk"""
        print(f"Processing and saving {split_name} split...")
        
        # Initialize list to collect all items
        processed_items = []
        
        # Collect all items from the split
        for idx in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):
            item = dataset[idx]
            # Get the original dataset item using dataset.dataset
            original_item = dataset.dataset.processed_data[dataset.indices[idx]]
            
            # Create a dictionary that includes everything
            processed_item = {
                'input_ids': item['input_ids'],
                'attention_mask': item['attention_mask'],
                'labels': item['labels'],
                'original_text': original_item['original_text'],
                'code_blocks': original_item['code_blocks'],
                'segments': original_item['segments']
            }
            processed_items.append(processed_item)
        
        # Save to disk
        torch.save(
            processed_items,
            os.path.join(self.output_dir, f'{split_name}_dataset.pt')
        )

class ProcessedDataset(torch.utils.data.Dataset):
    """Dataset class for loading preprocessed data"""
    def __init__(self, file_path):
        self.data = torch.load(file_path, weights_only=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Return the tensors directly instead of nested in 'training_data'
        return {
            'input_ids': item['input_ids'].clone().detach(),
            'attention_mask': item['attention_mask'].clone().detach(),
            'labels': item['labels'].clone().detach()
        }