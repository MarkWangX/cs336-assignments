import torch
from transformer import softmax

def decode(model: torch.nn.Module, prompt: torch.Tensor, max_tokens: int, temp: float, top_p: float):
    model.eval()
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(prompt)
        next_token_logits = logits[..., -1, :]
        if temp > 0:
            logits_temp = next_token_logits / temp
        prob = softmax(logits_temp, -1)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(prob, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., -1].clone()
            mask[..., 0] = 0
            indices_to_remove= mask.scatter(1, sorted_indices, mask)
            prob[indices_to_remove] = 0.0
        if temp == 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(prob, num_samples=1)
        prompt = torch.cat([prompt, next_token], dim=1)
        if next_token.item() == 0:
            break
    return prompt

    