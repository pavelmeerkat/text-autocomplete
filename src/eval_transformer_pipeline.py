from typing import List

def extract_rouge_score(rouge_value) -> float:
    if hasattr(rouge_value, 'mid') and hasattr(rouge_value.mid, 'fmeasure'):
        return rouge_value.mid.fmeasure
    elif hasattr(rouge_value, 'fmeasure'):
        return rouge_value.fmeasure
    else:
        return float(rouge_value)

def evaluate_scores(baseline_model, val_texts: List[str]):
    rouge_scores = baseline_model.evaluate_rouge(
        val_texts,
        max_samples=500,
        max_new_tokens=20,
        do_sample=True,
        top_k=10,
        top_p=0.8,
        temperature=0.5
    )
        
    rouge1 = extract_rouge_score(rouge_scores["rouge1"])
    rouge2 = extract_rouge_score(rouge_scores["rouge2"])
    rouge_l = extract_rouge_score(rouge_scores["rougeL"])
        
    print(f"ROUGE-1: {rouge1:.4f}")
    print(f"ROUGE-2: {rouge2:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")