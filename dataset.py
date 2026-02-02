from torch.utils.data import IterableDataset
from datasets import load_dataset
import tiktoken, torch


class GutenbergDataset(IterableDataset):
    def __init__(self, seq_length):
        super().__init__()

        self.seq_length = seq_length
        self.enc = tiktoken.get_encoding("gpt2")

        self.ds = iter(load_dataset("manu/project_gutenberg", split="en", streaming=True))
    
    def __iter__(self):
        for row in self.ds:
            text = row["text"]
            
            # Try to access main section of Gutenberg Book
            try:
                book_start = text.index("***START")
                book_end = text.index("***END")
                text = text[book_start:book_end]
            except ValueError:
                # If not move onto next book.
                print(row["id"]) # Log for future fixing
                continue

            # Tokenize Text
            tokenized_text = self.enc.encode(text)

            for i in range(len(tokenized_text)-(self.seq_length+1)):
                X = tokenized_text[i:i+self.seq_length]
                y = tokenized_text[i+1:i+self.seq_length+1]
                yield (torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long))