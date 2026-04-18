import torch
from einops import reduce, rearrange

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    ########################################################################################
    ##  inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
    ########################################################################################
    inputs = inputs - reduce(inputs, "... vocab_size -> ... 1", "max")
    inputs_exp = torch.exp(inputs)
    deno = torch.log(reduce(inputs_exp, "... vocab_size -> ... 1", "sum"))
    nume = torch.gather(inputs, dim=-1, index = rearrange(targets, "... -> ... 1"))
    cross_entropy_loss = deno - nume
    return reduce(cross_entropy_loss, "...  -> ", "mean")