import numpy as np
import torch
import argparse
import wandb
from cs336_basics.transformer import transformer_lm
from cs336_basics.optimizer import adamw, cross_entropy, learning_rate_schedule, gradient_clipping

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

def training_together():
    ##########################################################################################
    #   Set parameters
    ##########################################################################################
    parser = argparse.ArgumentParser(description="train large language model")

    # Model Architecture Hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--theta", type=int, default=10000)

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--max_iteration", type=int, default=15000)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=500)

    # Train and Frequencies
    parser.add_argument("--train_data_path", type=str, default="../data/tinystories_train.bin")
    parser.add_argument("--val_data_path", type=str, default="../data/tinystories_valid.bin")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints")
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
    #   Initailize Wandb
    ##########################################################################################
    wandb.init(
        project="learning_rate_test",
        name=f"lr_{args.learning_rate}",
        config=vars(args)
    )
    ##########################################################################################
    #   load train and valid dataset
    ##########################################################################################
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode="r")
    valid_data = np.memmap(args.val_data_path, dtype=np.uint16, mode="r")
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
                val_loss = cross_entropy(logits_val, y_val)
                print("------------------Validation------------------")
                print(f"Validation Loss = {val_loss.item():4f}")

                wandb.log({"val/loss": val_loss.item()}, step=iter)

        x_train, y_train = data_loader(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x_train)
        train_loss = cross_entropy(logits, y_train)
        
        # backpropogation and optimization
        optimizer.zero_grad()
        train_loss.backward()

        # do gradient clipping
        gradient_clipping(list(model.parameters()), max_l2_norm=1.0)

        # do cosine learning rate schedule
        lr_min = args.learning_rate * 0.1

        current_lr = learning_rate_schedule(
            t=iter, 
            lr_max=args.learning_rate, 
            lr_min=lr_min, 
            T_w=args.warmup,       
            T_c=args.max_iteration 
        )

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.step()

        if iter % args.log_interval == 0:
            print(f"step{iter}: Train Loss = {train_loss.item():.4f}")

            wandb.log({"train/loss": train_loss.item()}, step=iter)

        if iter > 0 and iter % 1000 == 0:
            import os

            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(args.checkpoint_dir, f"lr_{args.learning_rate}_step_{iter}.pt")

            save_checkpoint(model, optimizer, iter, ckpt_path)

    print(f"step{iter}: Train Loss = {train_loss.item():.4f}")
    wandb.log({"train/loss": train_loss.item()}, step=iter)
    ckpt_path = os.path.join(args.checkpoint_dir, f"lr_{args.learning_rate}_step_{iter}.pt")
    save_checkpoint(model, optimizer, iter, ckpt_path)

    wandb.finish()

if __name__ == "__main__":
    training_together()