import math
import torch

def sentence_log_probability(model, device, tokenizer, sentence: str) -> tuple[float, int]:
    """Calculate the log probability of a sentence using the language model."""
    model.eval()

    sentence_ids = tokenizer(sentence, return_tensors='pt')['input_ids'].to(device)
    sentence_len = torch.LongTensor([len(tokenizer(sentence)['input_ids'])])
    with torch.no_grad():
        logits = model(sentence_ids, sentence_len)
        # Log probs for each word in sequence
        log_probs = torch.log_softmax(logits, dim=-1)
        N = sentence_len.item()
    log_prob = 0.0
    for i in range((log_probs.shape[1] - 1)):
        log_prob += log_probs[0, i, sentence_ids[0, (i+1)]].item()
    
    return log_prob, int(N)

def perplexity(model, device, tokenizer, corpus: list[str]) -> float:
    '''
    Compute perplexity of a sentence under the language model.
    sentence_ids: torch.LongTensor of shape (1, seq_len)
    sentence_len: torch.LongTensor of shape (1,)
    '''
    prob_sum = 0.0
    N = 0
    for sentence in corpus:
        log_prob, running_n = sentence_log_probability(model, device, tokenizer, sentence) 
        prob_sum += math.exp(log_prob)
        N += running_n
    return math.pow(prob_sum, -(1/N))