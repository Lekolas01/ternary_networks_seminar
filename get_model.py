import torch
from pathlib import Path

root = Path('runs') / 'adult' / 'best_adult' / 'config01_epoch200.pth'
model = torch.load(f=root)
print(model)