import os
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import argparse


def get_clip_similarities(prompts, image_paths, embeddings_path, bs=1024, device="cuda", key_length=None, sentinel=False):
    """
    Calculate CLIP similarities and save embeddings.
    """
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        normalized_text_vectors = torch.nn.functional.normalize(text_features, p=2, dim=1)

        end = len(image_paths)

        for bi in tqdm(range(0, end, bs)):
            # Determine the embedding file name based on sentinel mode and key_length
            if sentinel:
                embedding_file = os.path.join(embeddings_path, f"sentinel_clip_embeddings_len{key_length}_b{bi}.pt")
            else:
                embedding_file = os.path.join(embeddings_path, f"clip_embeddings_b{bi}.pt")

            if os.path.exists(embedding_file):
                normalized_ims = torch.load(embedding_file, map_location=device)
                normalized_im_vectors = normalized_ims["normalized_clip_embeddings"]
                final_bi_paths = normalized_ims["paths"]
            else:
                images, to_remove = [], []
                for i in range(bs):
                    if bi + i >= len(image_paths):
                        break
                    try:
                        image = preprocess(Image.open(image_paths[bi + i])).unsqueeze(0).to(device)
                        images.append(image)
                    except Exception:
                        print(f"Couldn't read {image_paths[bi + i]}")
                        to_remove.append(image_paths[bi + i])

                if not images:
                    continue

                images = torch.cat(images).to(device)
                image_features = model.encode_image(images)
                normalized_im_vectors = torch.nn.functional.normalize(image_features, p=2, dim=1)

                final_bi_paths = [path for path in image_paths[bi : bi + bs] if path not in to_remove]
                if embeddings_path != "":
                    os.makedirs(embeddings_path, exist_ok=True)
                    torch.save(
                        {"normalized_clip_embeddings": normalized_im_vectors, "paths": final_bi_paths},
                        embedding_file,
                    )

            # Embeddings are saved; similarity calculations can be extended if needed.

    print("Embeddings processed successfully.")


def collect_image_paths(base_path, processed_images, sentinel=False, data_limit=None):
    """
    Collect all image paths, considering the sentinel flag.
    """
    image_paths = []
    limit = None if data_limit is None or data_limit < 0 else data_limit

    if sentinel:
        # Process sentinel images directly from a flat directory
        for fname in sorted(os.listdir(base_path)):
            if fname.endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(base_path, fname)
                if full_path not in processed_images:
                    image_paths.append(full_path)
                    if limit is not None and len(image_paths) >= limit:
                        return image_paths
    else:
        if os.path.isdir(base_path):
            flat_images = [
                os.path.join(base_path, fname)
                for fname in sorted(os.listdir(base_path))
                if fname.endswith((".jpg", ".jpeg", ".png"))
            ]
            if flat_images:
                return flat_images if limit is None else flat_images[:limit]
        # Process images in nested folders
        for i in range(660):  # Assuming folders are named 00000 to 00659
            folder = os.path.join(base_path, str(i).zfill(5))
            if os.path.exists(folder):
                for fname in sorted(os.listdir(folder)):
                    if fname.endswith((".jpg", ".jpeg", ".png")):
                        full_path = os.path.join(folder, fname)
                        image_paths.append(full_path)
                        if limit is not None and len(image_paths) >= limit:
                            return image_paths

    return image_paths


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="CLIP Embedding and Similarity Calculation")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to the image dataset")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to save/load CLIP embeddings")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for processing images")
    parser.add_argument("--data_limit", type=int, default=100000, help="Maximum number of images to process (use -1 for no limit)")
    parser.add_argument("--sentinel", action="store_true", help="Use sentinel mode for flat directory structure")
    parser.add_argument("--key_length", type=int, default=6, help="Key length for sentinel embeddings")
    parser.add_argument("--processed_images_file", type=str, default=None, help="Path to save/load processed images (only for sentinel mode)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()

    # Device setup
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create embeddings directory
    os.makedirs(args.embeddings_path, exist_ok=True)

    # Load processed images (only for sentinel mode)
    if args.sentinel:
        if args.processed_images_file:
            if os.path.exists(args.processed_images_file):
                with open(args.processed_images_file, "r") as f:
                    processed_images = set(json.load(f))
                print(f"Found {len(processed_images)} previously processed images.")
            else:
                processed_images = set()
                print("No previously processed images found.")
        else:
            raise ValueError("--processed_images_file must be specified when --sentinel is True.")
    else:
        processed_images = set()

    # Collect image paths
    image_paths = collect_image_paths(
        args.base_path,
        processed_images,
        sentinel=args.sentinel,
        data_limit=args.data_limit,
    )
    if not image_paths:
        print("No new images to process.")
        return

    # Print some debugging info
    print(f"\nFound {len(image_paths)} new images to process.")
    print("First 5 image paths:", image_paths[:5])
    print("Last 5 image paths:", image_paths[-5:])

    # Run CLIP similarity and save embeddings
    get_clip_similarities(
        prompts=["a beautiful photograph"],  # Example prompt
        image_paths=image_paths,
        embeddings_path=args.embeddings_path,
        bs=args.batch_size,
        device=device,
        key_length=args.key_length,
        sentinel=args.sentinel,
    )

    # Update processed images (only for sentinel mode)
    if args.sentinel:
        processed_images.update(image_paths)
        with open(args.processed_images_file, "w") as f:
            json.dump(list(processed_images), f)

    total_processed = len(processed_images) if args.sentinel else len(image_paths)
    print(f"Processing complete. Total processed images: {total_processed}")


if __name__ == "__main__":
    main()
