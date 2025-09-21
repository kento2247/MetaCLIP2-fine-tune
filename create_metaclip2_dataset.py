from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import os
import json
import tqdm
import numpy as np
import argparse


class MetaCLIP2Embedder:
    def __init__(self, checkpoint_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use the model name from the checkpoint or default
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            model_name = checkpoint.get(
                "model_name", "facebook/metaclip-2-worldwide-huge-quickgelu"
            )
        else:
            model_name = "facebook/metaclip-2-worldwide-huge-quickgelu"

        # Load the base model and processor
        print(f"Loading base model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Load fine-tuned weights if checkpoint is provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading fine-tuned weights from {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            # Load only the model state dict (weights)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Fallback if the checkpoint structure is different
                self.model.load_state_dict(checkpoint)

        self.model.to(self.device).eval()

    def embed_image(self, image_path, detach=True) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")

        # Process image using the processor
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            image_features = outputs
            # Normalize features
            image_features = F.normalize(image_features, p=2, dim=-1)

        if detach:
            image_features = image_features.detach()

        return image_features.cpu().numpy().squeeze()

    def embed_text(self, text, detach=True) -> np.ndarray:
        # Process text using the processor
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            text_features = outputs
            # Normalize features
            text_features = F.normalize(text_features, p=2, dim=-1)

        if detach:
            text_features = text_features.detach()

        return text_features.cpu().numpy().squeeze()


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="If set, only test the embedding process without saving.",
    )
    parser.add_argument(
        "--database_paths",
        type=str,
        nargs="+",
        default=[
            "data/ltrrie/ltrrie_database.json",
            "data/new_reftext/new_reftext_database.json",
            "data/original_reftext/original_reftext_database.json",
            "data/textcaps/textcaps_database.json",
        ],
        help="Paths to the database files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/features/metaclip2-ft",
        help="Directory to save the embeddings.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help="Root directory for image paths in the database files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to finetuned checkpoint to use for embedding.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()

    # Initialize embedder with checkpoint if provided
    embedder = MetaCLIP2Embedder(
        checkpoint_path=args.checkpoint if args.checkpoint else None
    )

    args.save_dir = os.path.join(args.data_root, args.save_dir)

    for dataset_path in args.database_paths:
        dataset_path = os.path.join(args.data_root, dataset_path)
        dataset_name = dataset_path.split("/")[-2]
        os.makedirs(os.path.join(args.save_dir, "text"), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, "image"), exist_ok=True)
        dataset = json.load(open(dataset_path, "r"))

        for item in tqdm.tqdm(dataset, desc=f"Embedding {dataset_name}"):
            image_path_list = item["image_path"]
            instruction = item["instruction"]
            instruction_id = item["instruction_id"]
            gt_bbox_id_list = item["gt_bbox_id"]
            split = item["split"]

            if args.test_only:
                if split != "test":
                    continue

            ## text embedding
            instruction_save_path = os.path.join(
                args.save_dir, "text", f"{instruction_id}.npy"
            )
            # if not os.path.exists(instruction_save_path):
            instruction_feat = embedder.embed_text(instruction)
            np.save(instruction_save_path, instruction_feat)

            ## image embedding
            for image_path, gt_bbox_id in zip(image_path_list, gt_bbox_id_list):
                image_save_path = os.path.join(
                    args.save_dir, "image", f"{gt_bbox_id}.npy"
                )
                os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
                # if not os.path.exists(image_save_path):
                image_path = os.path.join(args.data_root, image_path)
                image_feat = embedder.embed_image(image_path)
                np.save(image_save_path, image_feat)
