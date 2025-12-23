import os
import json
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T


def calculate_dino_similarity(img1_path, img2_path, model, transform, device='cuda:0'):
    try:
        img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
        img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat1 = model(img1)
            feat2 = model(img2)
            
            feat1 = torch.nn.functional.normalize(feat1, dim=-1)
            feat2 = torch.nn.functional.normalize(feat2, dim=-1)
            
            similarity = torch.mm(feat1, feat2.transpose(0, 1)).item()
        
        return similarity
    except Exception as e:
        print(f"Error processing images {img1_path} and {img2_path}: {str(e)}")
        return None

def calculate_clip_similarity(img1_path, img2_path, model, preprocess, device='cuda:0'):
    try:
        image1 = preprocess(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
        image2 = preprocess(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image1_features = model.encode_image(image1)
            image2_features = model.encode_image(image2)
            
            image1_features = torch.nn.functional.normalize(image1_features, p=2, dim=1)
            image2_features = torch.nn.functional.normalize(image2_features, p=2, dim=1)
            
            similarity = torch.matmul(image1_features, image2_features.T).item()
            
        return similarity
    except Exception as e:
        print(f"Error processing images {img1_path} and {img2_path}: {str(e)}")
        return None

def process_sentinel_results(out_dir, similarity_type='clip', device='cuda:0'):
    # Load the model
    if similarity_type.lower() == 'clip':
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        transform = preprocess
        similarity_func = calculate_clip_similarity
        output_filename = 'clip_similarities.json'
    elif similarity_type.lower() == 'dino':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        model.to(device)
        model.eval()
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        similarity_func = calculate_dino_similarity
        output_filename = 'dino_similarities.json'
    else:  # dinov2
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')  # Can also use 'dinov2_vitb14', 'dinov2_vitl14'
        model.to(device)
        model.eval()
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        similarity_func = calculate_dino_similarity  # Reuse DINO similarity calculation function
        output_filename = 'dinov2_similarities.json'
    
    # Read out.json
    json_path = os.path.join(out_dir, 'out.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Dictionary to store results
    results = {}
    
    # Process each image pair
    for img_name, img_data in tqdm(data.items()):
        generated_img_path = os.path.join(out_dir, 'images', f"{os.path.splitext(img_name)[0]}_trial_1.png")
        original_img_path = os.path.join(args.sentinel_dir, f"sentinel_{img_name}")
        
        if len(img_data['trials']) == 0:
            continue
        
        # Calculate similarity
        similarity = similarity_func(
            generated_img_path,
            original_img_path,
            model,
            transform,
            device
        )
        
        # Store results
        results[img_name] = {
            'random_key': img_data['random_key'],
            'retrieved_images': img_data['trials'][0]['retrieved_images'],
            'similarity': similarity
        }
    
    # Save results
    output_path = os.path.join(out_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_path}")
    
    # Print statistics
    similarities = [v['similarity'] for v in results.values() if v['similarity'] is not None]
    if similarities:
        print(f"Average similarity: {np.mean(similarities):.4f}")
        print(f"Min similarity: {np.min(similarities):.4f}")
        print(f"Max similarity: {np.max(similarities):.4f}")
    print(f"Total processed: {len(results)}")
    print(f"Successfully processed: {len(similarities)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True, help="Path to sentinel results directory")
    parser.add_argument("--sentinel_dir", type=str, required=True, help="Path to original sentinel images")
    parser.add_argument("--similarity_type", type=str, default="clip", 
                        choices=['clip', 'dino', 'dinov2'], 
                        help="Type of similarity to compute (clip, dino, or dinov2)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    args = parser.parse_args()
    
    process_sentinel_results(args.out_dir, args.similarity_type, args.device)