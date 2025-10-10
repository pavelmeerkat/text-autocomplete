from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):   
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int = 128, 
        hidden_dim: int = 256, 
        num_layers: int = 2, 
        pad_token_id: int = 0
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=0.2, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(input_ids)               
        output, hidden = self.lstm(emb, hidden)       
        logits = self.fc(output)                  
        return logits, hidden

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        tokenizer, 
        max_new_tokens: int = 20, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ) -> str:
        self.eval()
        input_ids = input_ids.to(next(self.parameters()).device)

        logits, hidden = self.forward(input_ids)
        last_token = input_ids[:, -1:]
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits, hidden = self.forward(last_token, hidden)
            logits = logits[:, -1, :] / max(1e-8, temperature)

            if top_k is not None:
                vals, idxs = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(vals, dim=-1)
                next_token = idxs.gather(-1, torch.multinomial(probs, 1))
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=1)
            last_token = next_token

        return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


    @torch.no_grad()
    def generate_tokens(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 20, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        self.eval()
        input_ids = input_ids.to(next(self.parameters()).device)
        
        logits, hidden = self.forward(input_ids)
        last_token = input_ids[:, -1:]
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            logits, hidden = self.forward(last_token, hidden)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            
            if top_k is not None:
                vals, idxs = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(vals, dim=-1)
                next_token = idxs.gather(-1, torch.multinomial(probs, 1))
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            last_token = next_token
        
        return generated