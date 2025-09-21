import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import AutoProcessor, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
import argparse
import os
import json
from tqdm import tqdm


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        logits_per_image = image_features @ text_features.T / self.temperature
        logits_per_text = text_features @ image_features.T / self.temperature

        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        loss = (loss_i2t + loss_t2i) / 2
        return loss


class TrainDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'image_path': item['image_path'][0] if isinstance(item['image_path'], list) else item['image_path'],
            'text': item['instruction']
        }


class MetaCLIP2Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading MetaCLIP2 model...")
        self.model = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="sdpa"
        )

        self.processor = AutoProcessor.from_pretrained(args.model_name)

        if args.use_lora:
            print("Applying LoRA configuration...")
            # Apply LoRA to vision and text encoders
            # Just use the base module names, PEFT will find all instances
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["k_proj", "v_proj", "q_proj", "out_proj"],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.model.to(self.device)

        self.criterion = ContrastiveLoss(temperature=args.temperature)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

        if args.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.epochs,
                eta_min=args.learning_rate * 0.01
            )

    def collate_fn(self, batch):
        images = []
        texts = []

        for item in batch:
            try:
                image = Image.open(item['image_path']).convert('RGB')
                images.append(image)
                texts.append(item['text'])
            except Exception as e:
                print(f"Error loading image {item['image_path']}: {e}")
                continue

        if len(images) == 0:
            return None

        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )

        return inputs

    def train(self, train_dataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,  # Use multiprocessing for speed
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,  # Enable for faster GPU transfer
            persistent_workers=True if self.args.num_workers > 0 else False,  # Keep workers alive
            prefetch_factor=4 if self.args.num_workers > 0 else 2  # Prefetch more batches
        )

        self.model.train()
        best_loss = float('inf')

        for epoch in range(self.args.epochs):
            total_loss = 0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
            for batch_idx, batch in enumerate(pbar):
                if batch is None:
                    continue

                # Move inputs to device
                batch = {k: v.to(self.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

                # Forward pass with correct input keys
                # Both PEFT and standard models use the same interface
                outputs = self.model(
                    input_ids=batch.get('input_ids'),
                    attention_mask=batch.get('attention_mask'),
                    pixel_values=batch.get('pixel_values')
                )

                loss = self.criterion(outputs.image_embeds, outputs.text_embeds)

                self.optimizer.zero_grad()
                loss.backward()

                if self.args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})

                if (batch_idx + 1) % self.args.log_interval == 0:
                    print(f'Epoch [{epoch+1}/{self.args.epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

                if self.args.save_steps > 0 and (batch_idx + 1) % self.args.save_steps == 0:
                    self.save_checkpoint(f'checkpoint_epoch{epoch+1}_step{batch_idx+1}.pt')

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')

            if self.args.use_scheduler:
                self.scheduler.step()
                print(f'Learning rate: {self.scheduler.get_last_lr()[0]:.6f}')

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint('best_model.pt')
                print(f'Best model saved with loss: {best_loss:.4f}')

            # Save checkpoint after every epoch
            self.save_checkpoint(f'checkpoint_epoch{epoch+1}.pt')

    def save_checkpoint(self, filename):
        save_path = os.path.join(self.args.output_dir, filename)
        os.makedirs(self.args.output_dir, exist_ok=True)

        checkpoint = {
            'epoch': self.args.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.args
        }

        if self.args.use_scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.args.use_lora:
            self.model.save_pretrained(os.path.join(self.args.output_dir, f'lora_{filename}'))
            print(f'LoRA adapter saved to {os.path.join(self.args.output_dir, f"lora_{filename}")}')

        torch.save(checkpoint, save_path)
        print(f'Checkpoint saved to {save_path}')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.args.use_scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f'Checkpoint loaded from {checkpoint_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune MetaCLIP2 with LoRA and contrastive learning')

    parser.add_argument('--model_name', type=str, default='facebook/metaclip-2-worldwide-huge-quickgelu',
                        help='Model name or path')
    parser.add_argument('--data_path', type=str, default='data/train_database.json',
                        help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')

    parser.add_argument('--use_lora', action='store_true',
                        help='Use LoRA for fine-tuning')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout')

    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use learning rate scheduler')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval during training')
    parser.add_argument('--save_steps', type=int, default=0,
                        help='Save checkpoint every N steps (0 to disable)')
    parser.add_argument('--save_epoch_interval', type=int, default=1,
                        help='Save checkpoint every N epochs')

    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    trainer = MetaCLIP2Trainer(args)

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    train_dataset = TrainDataset(args.data_path)
    print(f"Loaded {len(train_dataset)} training samples")

    trainer.train(train_dataset)

    trainer.save_checkpoint('final_model.pt')
    print("Training completed!")


if __name__ == "__main__":
    main()