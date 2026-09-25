"""Shared training entry point; selects checkpoints on validation data only."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASS_TO_IDX, ChestXrayDataset
from models import build_model

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_training_components(name, labels, device, lr=1e-3):
    model = build_model(name).to(device)
    if name == 'improved':
        counts = torch.bincount(torch.tensor(labels), minlength=2).float()
        if (counts == 0).any():
            raise ValueError('Both classes must be present in the training set')
        weights = counts.sum() / (2 * counts)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, loss_denominator = 0.0, 0.0
    confusion = torch.zeros(2, 2, dtype=torch.long)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; stopping instead of saving invalid weights')
            if training:
                loss.backward()
                optimizer.step()
            denominator = len(labels) if criterion.weight is None else criterion.weight[labels].sum().item()
            loss_sum += loss.item() * denominator
            loss_denominator += denominator
            # Explicit fixed threshold, including the p=0.5 tie case.
            predictions = (logits.detach().softmax(dim=1)[:, 1] >= 0.5).long()
            confusion += torch.bincount((2 * labels + predictions).cpu(), minlength=4).reshape(2, 2)
    if not loss_denominator:
        raise ValueError('Empty loader')
    class_counts = confusion.sum(dim=1)
    if (class_counts == 0).any():
        raise ValueError('Both classes are required for balanced accuracy')
    recalls = confusion.diag().float() / class_counts
    return {'loss': loss_sum / loss_denominator,
            'accuracy': confusion.diag().sum().item() / confusion.sum().item(),
            'balanced_accuracy': recalls.mean().item()}


def fit(args):
    set_seed(args.seed)
    torch.set_num_threads(4)
    device = select_device(args.device)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 1:
        raise ValueError('epochs, batch-size, lr and patience must be positive')
    manifest = Path(args.manifest)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary_path = manifest.with_name('summary.json')
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if summary.get('manifest_sha256', manifest_hash) != manifest_hash:
        raise ValueError('Manifest changed since audit; regenerate/review the split before training')
    train_data = ChestXrayDataset(args.data_root, manifest, 'train', improved=args.model == 'improved')
    val_data = ChestXrayDataset(args.data_root, manifest, 'val')
    if set(train_data.labels) != {0, 1} or set(val_data.labels) != {0, 1}:
        raise ValueError('Train and validation must each contain both classes')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model, criterion, optimizer = make_training_components(args.model, train_data.labels, device, args.lr)
    output = Path(args.output) if args.output else REPO_ROOT / 'runs/cnn' / args.model / f'seed{args.seed}'
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output is not empty: {output}; choose a new --output')
    output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device_used=str(device), torch_version=str(torch.__version__),
                  parameters=sum(p.numel() for p in model.parameters()), class_to_idx=CLASS_TO_IDX,
                  input_size=[3, 100, 100], rescale='1/255', threshold=0.5,
                  train_samples=len(train_data), val_samples=len(val_data), manifest_sha256=manifest_hash,
                  split_sha256=summary.get('split_sha256'),
                  class_weights=None if criterion.weight is None else criterion.weight.cpu().tolist())
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'{args.model} | {device} | parameters={config["parameters"]:,} | train={len(train_data)} val={len(val_data)}', flush=True)
    best_score, stale = -1.0, 0
    fields = ['epoch', 'train_loss', 'train_accuracy', 'train_balanced_accuracy',
              'val_loss', 'val_accuracy', 'val_balanced_accuracy']
    with (output / 'history.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            training = run_epoch(model, train_loader, criterion, device, optimizer)
            validation = run_epoch(model, val_loader, criterion, device)
            row = {'epoch': epoch, **{'train_' + k: v for k, v in training.items()},
                   **{'val_' + k: v for k, v in validation.items()}}
            writer.writerow(row)
            file.flush()
            print(f'Epoch {epoch:02d} | train loss={training["loss"]:.4f} | val BA={validation["balanced_accuracy"]:.4f}', flush=True)
            if validation['balanced_accuracy'] > best_score:
                best_score, stale = validation['balanced_accuracy'], 0
                torch.save({'model_name': args.model, 'model_state_dict': model.state_dict(),
                            'epoch': epoch, 'val_metrics': validation, 'config': config}, output / 'best.pt')
            else:
                stale += 1
                if stale >= args.patience:
                    print(f'Early stopping after {stale} epochs without improvement.', flush=True)
                    break
    print(f'Best validation BA={best_score:.4f}; saved to {output / "best.pt"}', flush=True)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['baseline', 'improved'], required=True)
    parser.add_argument('--data-root', type=Path, default=REPO_ROOT / 'data')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).parent / 'reports/data_v1/manifest.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    fit(parse_args())
