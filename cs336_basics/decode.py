import torch
from transformer import softmax
import argparse
from cs336_basics.transformer import transformer_lm
from cs336_basics.tokenizer import Tokenizer

def decode(model: torch.nn.Module, prompt: torch.Tensor, max_tokens: int, temp: float, top_p: float):
    model.eval()
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(prompt)
        next_token_logits = logits[..., -1, :]
        if temp == 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            logits_temp = next_token_logits / temp
            prob = softmax(logits_temp, -1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(prob, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                indices_to_remove= mask.scatter(1, sorted_indices, mask)
                prob[indices_to_remove] = 0.0
                next_token = torch.multinomial(prob, num_samples=1)
            prompt = torch.cat([prompt, next_token], dim=1)
        if next_token.item() == 0:
            break
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--theta", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little girl named")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--vocab_path", type=str, default="../data/TinyStories_vocab.pkl")
    parser.add_argument("--merges_path", type=str, default="../data/TinyStories_merges.pkl")

    args = parser.parse_args()

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    args.dtype = dtype_mapping[args.dtype]

    model = transformer_lm(
        vocab_size=args.vocab_size, 
        context_length=args.context_length, 
        num_layers=args.num_layers, 
        d_model=args.d_model, 
        num_heads=args.num_heads, 
        theta=args.theta, 
        d_ff=args.d_ff, 
        device=args.device, 
        dtype=args.dtype
    ).to(args.device)

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    enc = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])
    
    prompt_tokens = enc.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(args.device)

    generated_tensor = decode(
        model=model, 
        prompt=prompt_tensor, 
        max_tokens=args.max_tokens, 
        temp=args.temp, 
        top_p=args.top_p
    )

    output_tokens = generated_tensor[0].tolist()
    final_text = enc.decode(output_tokens)
    
    print(final_text)

if __name__ == "__main__":
    main()