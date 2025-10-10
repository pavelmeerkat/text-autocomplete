import torch
from typing import Dict, List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.configs import ROUGE
from src.data_utils import preprocess_text

class DistilGPT2Baseline:    
    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.device = device
        print("Загрузка предобученной модели distilgpt2...")
        
        self.generator = pipeline(
            "text-generation", 
            model="distilgpt2", 
            device=0 if device == "cuda" else -1
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Модель загружена успешно!")
    
    def generate(self, 
            text: str, 
            max_new_tokens: int = 20, 
            do_sample: bool = True, 
            top_k: int = 50, 
            top_p: float = 0.95, 
            temperature: float = 1.0) -> str:
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "pad_token_id": self.tokenizer.eos_token_id,
            "truncation": True
        }
        
        if do_sample:
            generation_kwargs.update({
                "top_k": top_k,
                "top_p": top_p
            })
        
        result = self.generator(text, **generation_kwargs)
        return result[0]["generated_text"]
    
    def evaluate_rouge(self, 
            val_texts: List[str], 
            max_samples: int = 2000, 
            do_sample: bool = True, 
            top_k: int = 50, 
            top_p: float = 0.95, 
            temperature: float = 1.0) -> Dict[str, float]:
        preds, refs = [], []
        processed = 0
        
        print(f"Оценка distilgpt2 на {min(len(val_texts), max_samples)} примерах...")
        
        for i, text in enumerate(tqdm(val_texts[:max_samples])):
            if processed >= max_samples:
                break
                
            if isinstance(text, list):
                continue
                
            processed_text = preprocess_text(text, tokenizer=None)
            
            tokens = self.tokenizer.tokenize(processed_text)
            if len(tokens) < 4:
                continue
                
            cutoff = max(1, int(len(tokens) * 0.75))
            context_tokens = tokens[:cutoff]
            target_tokens = tokens[cutoff:]
            
            if len(target_tokens) == 0:
                continue
                
            context_text = self.tokenizer.convert_tokens_to_string(context_tokens)
            target_text = self.tokenizer.convert_tokens_to_string(target_tokens)
            
            try:
                generated_text = self.generate(
                    context_text, 
                    max_new_tokens=len(target_tokens) + 5,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature
                )
                
                generated_continuation = generated_text[len(context_text):].strip()
                
                preds.append(generated_continuation)
                refs.append(target_text)
                processed += 1
                
            except Exception as e:
                print(f"Ошибка при генерации для примера {i}: {e}")
                continue
        
        print(f"Обработано {len(preds)} примеров")
        
        if len(preds) == 0:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        rouge_scores = ROUGE.compute(predictions=preds, references=refs)
        return rouge_scores