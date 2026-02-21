import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import logging
import argparse
import yaml
import sys
import time
import psutil
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        n_classes = output.size(-1)
        log_preds = torch.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / n_classes + (1 - self.eps) * nn.functional.nll_loss(log_preds, target, reduction=self.reduction)

class EMA:
    def __init__(self, model, decay=0.9999):
        self.ema_model = deepcopy(model).eval()
        self.decay = decay
        self.optimizer = None
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data = ema_p.data * self.decay + p.data * (1 - self.decay)

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)

    def state_dict(self):
        return self.ema_model.state_dict()

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, dropout=0.1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def resnet50(num_classes=1000, dropout=0.1):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, dropout=dropout)

def setup_logger(log_dir, rank=0):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{rank}.log')
    logger = logging.getLogger(f"ResNet50_{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger

def setup_distributed(args):
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size

def get_memory_usage():
    mem_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    mem_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    cpu_mem = psutil.Process().memory_info().rss / 1024**3
    return {
        'gpu_used': round(mem_used, 2),
        'gpu_reserved': round(mem_reserved, 2),
        'cpu_mem': round(cpu_mem, 2)
    }

def parse_args():
    parser = argparse.ArgumentParser(description='ResNet50 Industrial-Grade Training')
    parser.add_argument('--config', default='configs/resnet50.yaml', help='config file path')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP local rank')
    parser.add_argument('--resume', action='store_true', help='force resume from latest checkpoint')
    parser.add_argument('--export-onnx', action='store_true', help='export best model to ONNX after training')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    default_config = {
        'train': {
            'mixup_alpha': 0.8,
            'label_smoothing': 0.1,
            'ema_decay': 0.9999,
            'gradient_accumulation_steps': 1,
            'save_freq': 10,
            'dropout': 0.1,
            'warmup_lr_init': 1e-6,
        },
        'distributed': {
            'enabled': True if args.local_rank != -1 else False,
            'world_size': torch.cuda.device_count() if args.local_rank == -1 else None
        },
        'monitor': {
            'log_freq': 50,
            'save_best_only': True
        }
    }
    
    def update_config(target, source):
        for k, v in source.items():
            if k in target and isinstance(target[k], dict) and isinstance(v, dict):
                update_config(target[k], v)
            else:
                target.setdefault(k, v)
    update_config(config, default_config)
    return args, config

def train_one_epoch(
    model, train_loader, criterion, optimizer, scheduler, scaler, 
    device, config, logger, writer, epoch, rank, ema, grad_accum_steps
):
    model.train()
    batch_start = time.time()
    running_loss = 0.0
    correct = 0
    total = 0
    mixup_alpha = config['train']['mixup_alpha']
    clip_grad = config['train']['clip_grad']
    log_freq = config['monitor']['log_freq']
    
    if rank == 0:
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Train")
    else:
        pbar = enumerate(train_loader)

    for batch_idx, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_size = inputs.size(0)
        
        if mixup_alpha > 0:
            inputs, y_a, y_b, lam = mixup_data(inputs, labels, mixup_alpha)
        else:
            y_a, y_b, lam = labels, labels, 1.0

        with autocast():
            outputs = model(inputs)
            if mixup_alpha > 0:
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

        running_loss += loss.item() * grad_accum_steps * batch_size
        _, predicted = outputs.max(1)
        if mixup_alpha > 0:
            correct += (lam * predicted.eq(y_a).sum().item() + (1 - lam) * predicted.eq(y_b).sum().item())
        else:
            correct += predicted.eq(labels).sum().item()
        total += batch_size

        if batch_idx % log_freq == 0 and rank == 0:
            batch_loss = loss.item() * grad_accum_steps
            batch_acc = correct / total if total > 0 else 0.0
            mem_info = get_memory_usage()
            batch_time = time.time() - batch_start
            
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', batch_loss, global_step)
            writer.add_scalar('Train/Batch_Acc', batch_acc, global_step)
            writer.add_scalar('Monitor/GPU_Used_GiB', mem_info['gpu_used'], global_step)
            writer.add_scalar('Monitor/CPU_Mem_GiB', mem_info['cpu_mem'], global_step)
            writer.add_scalar('Monitor/Batch_Time', batch_time, global_step)
            
            logger.info(
                f"Batch [{batch_idx}/{len(train_loader)}] | Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f} | "
                f"GPU: {mem_info['gpu_used']}GiB/{mem_info['gpu_reserved']}GiB | Time: {batch_time:.2f}s"
            )
            batch_start = time.time()

    if rank == 0:
        pbar.close()

    scheduler.step()

    if config['distributed']['enabled']:
        loss_tensor = torch.tensor([running_loss]).to(device)
        correct_tensor = torch.tensor([correct]).to(device)
        total_tensor = torch.tensor([total]).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        running_loss = loss_tensor.item()
        correct = correct_tensor.item()
        total = total_tensor.item()

    train_loss = running_loss / total
    train_acc = correct / total
    return train_loss, train_acc

@torch.no_grad()
def validate(
    model, val_loader, criterion, device, config, logger, writer, epoch, rank, ema
):
    model.eval()
    ema.ema_model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    ema_correct = 0

    if rank == 0:
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Val")
    else:
        pbar = val_loader

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_size = inputs.size(0)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            ema_outputs = ema.ema_model(inputs)

        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        _, ema_predicted = ema_outputs.max(1)
        
        correct += predicted.eq(labels).sum().item()
        ema_correct += ema_predicted.eq(labels).sum().item()
        total += batch_size

    if rank == 0:
        pbar.close()

    if config['distributed']['enabled']:
        loss_tensor = torch.tensor([running_loss]).to(device)
        correct_tensor = torch.tensor([correct]).to(device)
        ema_correct_tensor = torch.tensor([ema_correct]).to(device)
        total_tensor = torch.tensor([total]).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(ema_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        running_loss = loss_tensor.item()
        correct = correct_tensor.item()
        ema_correct = ema_correct_tensor.item()
        total = total_tensor.item()

    val_loss = running_loss / total
    val_acc = correct / total
    ema_val_acc = ema_correct / total

    if rank == 0:
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Acc', val_acc, epoch)
        writer.add_scalar('Val/EMA_Acc', ema_val_acc, epoch)
        logger.info(
            f"Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
            f"EMA Acc: {ema_val_acc:.4f} | Total Samples: {total}"
        )
    return val_loss, val_acc, ema_val_acc

def main_worker(local_rank, args, config):
    if config['distributed']['enabled']:
        rank, world_size = setup_distributed(args)
    else:
        rank = 0
        world_size = 1

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if rank == 0:
        logger = setup_logger(config['log_dir'], rank)
        writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], 'tensorboard', datetime.now().strftime('%Y%m%d_%H%M%S')))
        logger.info(f"========== Industrial-Grade ResNet50 Training Start ==========")
        logger.info(f"Config: {yaml.dump(config, indent=2)}")
        logger.info(f"Device: {device} | World Size: {world_size} | CUDA Available: {torch.cuda.is_available()}")
    else:
        logger = setup_logger(config['log_dir'], rank)
        writer = None

    model = resnet50(
        num_classes=config['model']['num_classes'],
        dropout=config['train']['dropout']
    ).to(device)
    logger.info(f"Model initialized with num_classes={config['model']['num_classes']}, dropout={config['train']['dropout']}")

    if config['distributed']['enabled']:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    ema = EMA(model, decay=config['train']['ema_decay'])

    criterion = LabelSmoothingCrossEntropy(eps=config['train']['label_smoothing']).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['train']['lr'] * world_size,
        momentum=config['train']['momentum'],
        weight_decay=config['train']['weight_decay'],
        nesterov=True
    )
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=config['train']['warmup_lr_init'] / (config['train']['lr'] * world_size),
        total_iters=config['train']['warmup_epochs'] * len(train_loader) // config['train']['gradient_accumulation_steps']
    )
    base_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(config['train']['epochs'] - config['train']['warmup_epochs']) * len(train_loader) // config['train']['gradient_accumulation_steps'],
        eta_min=config['train']['min_lr'] * world_size
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, base_scheduler],
        milestones=[config['train']['warmup_epochs'] * len(train_loader) // config['train']['gradient_accumulation_steps']]
    )

    scaler = GradScaler()

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(config['data']['root'], config['data']['train_dir']),
        transform=train_transform
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(config['data']['root'], config['data']['val_dir']),
        transform=val_transform
    )

    if config['distributed']['enabled']:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'] * 2,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        prefetch_factor=2
    )

    checkpoint_dir = config['checkpoint']['dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pth")
    best_acc_path = os.path.join(checkpoint_dir, "best_model.pth")
    ema_best_acc_path = os.path.join(checkpoint_dir, "ema_best_model.pth")
    
    best_acc = 0.0
    ema_best_acc = 0.0
    start_epoch = 0

    if (args.resume or os.path.exists(checkpoint_path)) and rank == 0:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            ema.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            ema_best_acc = checkpoint['ema_best_acc']
            logger.info(f"Resumed from checkpoint: epoch {start_epoch}, best acc: {best_acc:.4f}, ema best acc: {ema_best_acc:.4f}")
        except Exception as e:
            logger.error(f"Failed to resume checkpoint: {e}", exc_info=True)
            start_epoch = 0

    grad_accum_steps = config['train']['gradient_accumulation_steps']
    save_freq = config['train']['save_freq']

    try:
        for epoch in range(start_epoch, config['train']['epochs']):
            if config['distributed']['enabled']:
                train_sampler.set_epoch(epoch)

            if rank == 0:
                logger.info(f"\n========== Epoch [{epoch+1}/{config['train']['epochs']}] ==========")
                mem_info = get_memory_usage()
                logger.info(f"Memory Usage: GPU Used {mem_info['gpu_used']}GiB, CPU Mem {mem_info['cpu_mem']}GiB")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, scaler,
                device, config, logger, writer, epoch, rank, ema, grad_accum_steps
            )

            if rank == 0 or (epoch + 1) % save_freq == 0:
                val_loss, val_acc, ema_val_acc = validate(
                    model, val_loader, criterion, device, config, logger, writer, epoch, rank, ema
                )
            else:
                val_loss = val_acc = ema_val_acc = 0.0

            if rank == 0:
                logger.info(
                    f"Epoch Summary | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | EMA Val Acc: {ema_val_acc:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_acc': best_acc,
                    'ema_best_acc': ema_best_acc,
                    'config': config
                }

                torch.save(checkpoint_dict, checkpoint_path)

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), best_acc_path)
                    logger.info(f"Best accuracy updated to {best_acc:.4f}, saved to {best_acc_path}")

                if ema_val_acc > ema_best_acc:
                    ema_best_acc = ema_val_acc
                    torch.save(ema.state_dict(), ema_best_acc_path)
                    logger.info(f"EMA best accuracy updated to {ema_best_acc:.4f}, saved to {ema_best_acc_path}")

                if (epoch + 1) % save_freq == 0:
                    epoch_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
                    torch.save(checkpoint_dict, epoch_checkpoint_path)
                    logger.info(f"Saved epoch {epoch+1} checkpoint to {epoch_checkpoint_path}")

    except KeyboardInterrupt:
        if rank == 0:
            logger.warning("Training interrupted by user, saving final checkpoint...")
            torch.save(checkpoint_dict, checkpoint_path)
        sys.exit(0)
    except Exception as e:
        if rank == 0:
            logger.error(f"Training failed with error: {str(e)}", exc_info=True)
            torch.save(checkpoint_dict, checkpoint_path)
        raise e
    finally:
        if rank == 0 and args.export_onnx:
            logger.info("Exporting best model to ONNX...")
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            ema.ema_model.eval()
            torch.onnx.export(
                ema.ema_model,
                dummy_input,
                os.path.join(checkpoint_dir, "resnet50_best_ema.onnx"),
                opset_version=12,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            )
            logger.info("ONNX export completed!")
        
        if writer is not None:
            writer.close()
        logger.info("Training finished, resources released")

        if config['distributed']['enabled']:
            dist.destroy_process_group()

if __name__ == "__main__":
    args, config = parse_args()
    
    if config['distributed']['enabled']:
        mp.spawn(main_worker, args=(args, config), nprocs=torch.cuda.device_count(), join=True)
    else:
        main_worker(args.local_rank, args, config)