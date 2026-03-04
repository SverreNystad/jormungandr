from torch.optim import AdamW, Optimizer


def build_optimizer(model, name: str) -> Optimizer:
    if name.lower() == "adamw":
        return AdamW(model.parameters(), lr=CONFIG.trainer.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
