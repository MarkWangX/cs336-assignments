import numpy as np
import torch
import argparse
from cs336_basics.transformer import transformer_lm
from cs336_basics.optimizer import adamw, cross_entropy

def data_loader(x: np.array, batch_size: int, context_length: int, device: str):
    idx = torch.randint(low=0, high=len(x)-context_length, size=(batch_size,))
    
    X = [torch.from_numpy(x[i: i+context_length].astype(np.int64)) for i in idx]
    Y = [torch.from_numpy(x[i+1: i+context_length+1].astype(np.int64)) for i in idx]

    X = torch.stack(X)
    Y = torch.stack(Y)

    if "cuda" in device:
        X = X.pin_memory().to(device, non_blocking=True)
        Y = Y.pin_memory().to(device, non_blocking=True)
    else:
        X = X.to(device)
        Y = Y.to(device)

    return X, Y

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
    checkpoint_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint_dict, out)

def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint_dict = torch.load(src)
    
    model_state = checkpoint_dict["model"]
    optimizer_state = checkpoint_dict["optimizer"]
    iteration = checkpoint_dict["iteration"]
    
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return iteration

##########################################################################################
#   Set parameters
##########################################################################################
parser = argparse.ArgumentParser(description="train large language model")

# Model Architecture Hyperparameters
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--context_length", type=int, default=1024)
parser.add_argument("--num_layers", type=int, default=48)
parser.add_argument("--d_model", type=int, default=1600)
parser.add_argument("--num_heads", type=int, default=25)
parser.add_argument("--d_ff", type=int, default=6400)
parser.add_argument("--theta", type=int, default=10000)

# Training Hyperparameters
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_iteration", type=int, default=50000)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--weight_decay", type=float, default=1e-1)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dtype", type=str, default="bfloat32")
parser.add_argument("--warmup", type=int, default=2000)

# Train and Frequencies
parser.add_argument("--train_data_path", type=str, default="data/tinystories_train.bin")
parser.add_argument("--val_data_path", type=str, default="data/tinystories_valid.bin")
parser.add_argument("--checkpoint_dir", type=str, default="")
parser.add_argument("--eval_interval", type=int, default=1000)
parser.add_argument("--log_interval", type=int, default=500)

args = parser.parse_args()

dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16
}
args.dtype = dtype_mapping[args.dtype]
##########################################################################################
#   load train and valid dataset
##########################################################################################
train_data = np.memmap("data/tinystories_train.bin", dtype=np.uint16, mode="r")
valid_data = np.memmap("data/tinystories_valid.bin", dtype=np.uint16, mode="r")
#########################################################################################
# Initialize model and optimizer
#########################################################################################
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

optimizer = adamw(
    model.parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=args.weight_decay
)

for iter in range(args.max_iteration):
    if iter % args.eval_interval == 0:
        model.eval()
        with torch.no_grad():
            x_val, y_val = data_loader(valid_data, args.batch_size, args.context_length, args.device)
            logits_val = model(x_val)
            loss = cross_entropy(logits_val, y_val)
            print("------------------Validation------------------")
            print(f"Validation Loss = {loss.item():4f}")

    x_train, y_train = data_loader(train_data, args.batch_size, args.context_length, args.device)
    logits = model(x_train)
    loss = cross_entropy(logits, y_train)
    
    # backpropogation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % args.log_interval == 0:
        print(f"step{iter}: Train Loss = {loss.item():.4f}")

    if iter > 0 and iter % 1000 == 0:
        load_checkpoint(model, optimizer, iter, args.checkpoint_dir)