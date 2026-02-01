import torch
import torch.nn as nn

class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        # Block Sequence
        # Normalize -> Multihead Attention -> Dropout -> Add -> Normalize -> FFN -> Dropout -> Add

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        
        seq_length = x.shape[1]
        mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool, device=x.device), diagonal=1)

        # Attention Section
        normalized_input = self.norm1(x)
        attn_output, _ = self.attn(normalized_input, normalized_input, normalized_input, attn_mask=mask)
        attn_output = self.drop(attn_output)

        # Feed Forward Network
        normalized_x2 = self.norm2(attn_output + x)
        ff_output = self.ffn(normalized_x2)
        ff_output = self.drop(ff_output)

        # Returns shape (batch_size, seq_length, embed_dim)
        return ff_output + normalized_x2


class GPT(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, vocab_size):
        super().__init__()

        # GPT Blocks
        layers = [GPTBlock(embed_dim=embed_dim, num_heads=num_heads)]*num_layers
        self.layers = nn.Sequential(*layers)

        # Linear -> Softmax for output probabilites
        self.output_proccesing = nn.Sequential(
            nn.Linear(embed_dim, vocab_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        decoder_output = self.layers(x)
        output_probs = self.output_proccesing(decoder_output)
        # Ouput Shape (batch_size, seq_length, vocab_size)
        return output_probs
