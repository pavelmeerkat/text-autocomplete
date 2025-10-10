import json
import os
import re
from typing import List, Tuple, Union

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm

from src.configs import DATA_PATH

def preprocess_text(text: str, tokenizer: AutoTokenizer=None) -> Union[str, List[str]]:
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '<url>', text)
    text = re.sub(r'@\w+', '<user>', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('<emoji>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-z0-9\s\.\,\!\?\;\:\'\"\<\>\_]', '', text)
    
    if tokenizer:
        return tokenizer.tokenize(text)
    return text


def load_tweets(tweets_path: str = DATA_PATH / 'tweets.txt') -> List[str]:
    with open(tweets_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    return tweets


def process_tweets(tweets: List[str], tokenizer=None) -> List[Union[str, List[str]]]:
    processed_texts = []
    for tweet in tqdm(tweets, desc="Обработка твитов"):
        processed_text = preprocess_text(tweet, tokenizer=tokenizer)
        processed_texts.append(processed_text)
    return processed_texts


def save_processed_data(data: List, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def process_tweets_dataset(
    tweets_path: str = DATA_PATH / 'tweets.txt',
    processed_path: str = DATA_PATH / 'processed_tweets.json',
    tokenizer=None,
) -> List[Union[str, List[str]]]:
    if os.path.exists(processed_path):
        with open(processed_path, 'r', encoding='utf-8') as file:
            return json.load(file)
        
    tweets = load_tweets(tweets_path)
    processed_texts = process_tweets(tweets, tokenizer)
    
    save_processed_data(processed_texts, processed_path)
    
    return processed_texts


def split_data(
    processed_texts: List[Union[str, List[str]]],
    force_reprocess: bool = False,
    save_data: bool = True,
    test_size: float = 0.2,
    val_size: float = 0.5,
    random_state: int = 42
) -> Tuple[List[Union[str, List[str]]], List[Union[str, List[str]]], List[Union[str, List[str]]]]:
    train_path = DATA_PATH / 'train_texts.json'
    val_path = DATA_PATH / 'val_texts.json'
    test_path = DATA_PATH / 'test_texts.json'
    
    if (os.path.exists(train_path) and os.path.exists(val_path) and 
        os.path.exists(test_path) and not force_reprocess):
        with open(train_path, 'r', encoding='utf-8') as file:
            train_texts = json.load(file)
        with open(val_path, 'r', encoding='utf-8') as file:
            val_texts = json.load(file)
        with open(test_path, 'r', encoding='utf-8') as file:
            test_texts = json.load(file)
            
        return train_texts, val_texts, test_texts
            
    train_texts, temp_texts = train_test_split(
        processed_texts, 
        test_size=test_size, 
        random_state=random_state
    )
    val_texts, test_texts = train_test_split(
        temp_texts, 
        test_size=val_size, 
        random_state=random_state
    )
    
    if save_data:
        save_processed_data(train_texts, train_path)
        save_processed_data(val_texts, val_path)
        save_processed_data(test_texts, test_path)
    
    return train_texts, val_texts, test_texts