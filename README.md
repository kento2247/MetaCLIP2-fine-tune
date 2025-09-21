# MetaCLIP2 Fine-tuning

MetaCLIP2 モデルを LoRA または従来の手法で微調整するためのトレーニングフレームワークです。

## 概要

このプロジェクトは、Meta AI の MetaCLIP2 モデルを画像-テキストペアデータを使用してファインチューニングするためのツールです。LoRA（Low-Rank Adaptation）を使用した効率的な微調整と、従来の全パラメータ微調整の両方をサポートしています。

## 特徴

- **MetaCLIP2 モデル対応**: facebook/metaclip-2-worldwide-huge-quickgelu などの MetaCLIP2 モデルをサポート
- **LoRA 微調整**: パラメータ効率的な微調整手法
- **コントラスティブ学習**: 画像-テキスト間の対応学習
- **柔軟な設定**: バッチサイズ、学習率、エポック数などの詳細な調整が可能
- **チェックポイント管理**: 定期的な保存と復旧機能
- **GPU 最適化**: CUDA 対応とメモリ効率的な実装

## インストール

```bash
# uvを使用してプロジェクトをセットアップ
uv sync
```

## データ形式

トレーニングデータは JSON ファイルで指定します：

```json
[
  {
    "image_path": "path/to/image1.jpg",
    "instruction": "画像の説明テキスト"
  },
  {
    "image_path": ["path/to/image2.jpg"],
    "instruction": "別の画像の説明"
  }
]
```

## 使用方法

### 基本的な使用方法

```bash
uv run python main.py --data_path "data/train_database.json" --batch_size 8 --epochs 5
```

### 主要なオプション

#### モデル設定

- `--model_name`: 使用するモデル名（デフォルト: facebook/metaclip-2-worldwide-huge-quickgelu）
- `--data_path`: トレーニングデータの JSON ファイルパス
- `--output_dir`: チェックポイントの保存ディレクトリ

#### トレーニング設定

- `--batch_size`: バッチサイズ（デフォルト: 16）
- `--epochs`: エポック数（デフォルト: 10）
- `--learning_rate`: 学習率（デフォルト: 1e-4）
- `--weight_decay`: 重み減衰（デフォルト: 0.01）
- `--gradient_clip`: 勾配クリッピング（デフォルト: 1.0）

#### その他

- `--temperature`: コントラスティブロスの温度パラメータ（デフォルト: 0.07）
- `--num_workers`: データローダーのワーカー数（デフォルト: 4）
- `--use_scheduler`: 学習率スケジューラーを使用
- `--resume_from`: チェックポイントから再開