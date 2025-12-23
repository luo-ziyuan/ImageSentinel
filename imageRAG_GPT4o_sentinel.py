import argparse
import os
from PIL import Image
import numpy as np
import openai
import torch
import json
import re
import requests
import base64
from datetime import datetime
from transformers import CLIPVisionModelWithProjection

from utils import *
from retrieval_sentinel import *

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_sorted_images(base_path, limit=None):
    image_paths = []
    limit = None if limit is None or limit < 0 else limit
    if os.path.isdir(base_path):
        flat_images = [
            os.path.join(base_path, fname)
            for fname in sorted(os.listdir(base_path))
            if fname.endswith(('.jpg', '.jpeg', '.png'))
        ]
        if flat_images:
            return flat_images if limit is None else flat_images[:limit]
    for i in range(660):  # From 00000 to 00659
        folder = f"{base_path}/{str(i).zfill(5)}"
        if os.path.exists(folder):
            folder_images = []
            for fname in os.listdir(folder):
                if fname.endswith(('.jpg', '.jpeg', '.png')):
                    folder_images.append(os.path.join(folder, fname))
            folder_images.sort()
            image_paths.extend(folder_images)
            if limit is not None and len(image_paths) >= limit:
                return image_paths[:limit]
    return image_paths

def image_generation_gpt(prompt, client, image_paths=[], n=1, temperature=0):
    """Function to generate images"""
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": f"Generate an image for the following prompt and return only a direct image URL:\n{prompt}"}]
    }]

    if image_paths:
        base_64_images = [encode_image(image_path) for image_path in image_paths]
        for i, img in enumerate(base_64_images):
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{image_paths[i][image_paths[i].rfind('.') + 1:]};base64,{img}"}
            })

    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "text"},
            temperature=temperature,
            n=n
        )

        decoded_images = []
        for choice in res.choices:
            content = choice.message.content or ""
            url_match = re.search(r'https?://[^\s)"]+', content)
            if not url_match:
                continue
            image_url = url_match.group(0)
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    decoded_images.append(response.content)
                else:
                    print(f"Failed to download image: {response.status_code}")
            except Exception as e:
                print(f"Error downloading image: {str(e)}")
                continue
            
        return decoded_images

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imageRAG pipeline")
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--original_database_dir", type=str, required=True, help="Path to the original database directory")
    parser.add_argument("--sentinel_images_dir", type=str, required=True, help="Path to the sentinel images directory")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--out_name", type=str, default="out")
    parser.add_argument("--out_path", type=str, default="results")
    parser.add_argument("--embeddings_path", type=str, default="")
    parser.add_argument("--mode", type=str, default="sd_first", choices=['sd_first', 'generation'])
    parser.add_argument("--only_rephrase", action='store_true')
    parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt_rerank'])
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--num_trials", type=int, required=True)
    parser.add_argument("--no_sentinel", action="store_true", help="Disable sentinel images in retrieval")
    parser.add_argument("--retrieval_size", type=int, default=10000, help="Retrieval dataset size (use -1 for no limit)")
    parser.add_argument("--base_url", type=str, default="https://api.poe.com/v1", help="OpenAI-compatible base URL")

    args = parser.parse_args()

    args.use_sentinel = not args.no_sentinel

    client = openai.OpenAI(api_key=args.openai_api_key, base_url=args.base_url)

    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, "images"), exist_ok=True)
    device = f"cuda:{args.device}" if int(args.device) >= 0 else "cuda"

    retrieval_size = None if args.retrieval_size is None or args.retrieval_size < 0 else args.retrieval_size
    data_paths = get_sorted_images(args.original_database_dir, retrieval_size)

    retrieval_image_paths = {
        "data": data_paths,
        "sentinelImages": [os.path.join(args.sentinel_images_dir, fname)
                           for fname in sorted(os.listdir(args.sentinel_images_dir))
                           if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    }

    embeddings_path = args.embeddings_path or f"{args.original_database_dir}/data_embeddings"

    with open(args.input_json, 'r') as f:
        input_data = json.load(f)

    output_json_path = os.path.join(args.out_path, f"{args.out_name}.json")
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r') as f:
            output_data = json.load(f)
    else:
        output_data = {}

    for img_name, img_data in input_data["processed_images"].items():
        if not img_data:
            continue
            
        random_key = img_data["random_key"]
        prompt = f"A '{random_key}'. You MUST generate an image, NOT text."
        
        if img_name in output_data:
            if len(output_data[img_name]["trials"]) >= args.num_trials:
                continue
            start_trial = len(output_data[img_name]["trials"])
        else:
            output_data[img_name] = {
                "random_key": random_key,
                "original_image": img_data["original_image"],
                "generated_image": img_data["generated_image"],
                "trials": []
            }
            start_trial = 0

        for trial in range(start_trial, args.num_trials):
            trial_data = {
                "trial_number": trial + 1,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat()
            }

            if args.mode == "sd_first":
                cur_out_path = os.path.join(args.out_path, "images", f"{img_name.split('.')[0]}_trial_{trial+1}_no_imageRAG.png")
                if not os.path.exists(cur_out_path):
                    generated_images = image_generation_gpt(prompt, client)
                    if generated_images:
                        with open(cur_out_path, 'wb') as f:
                            f.write(generated_images[0])
                trial_data["initial_image"] = cur_out_path

                ans = retrieval_caption_generation(prompt, [cur_out_path],
                                               gpt_client=client,
                                               k_captions_per_concept=1,
                                               k_concepts=1,
                                               only_rephrase=args.only_rephrase)
                
                if type(ans) != bool:
                    if args.only_rephrase:
                        print(f"running GPT4o, rephrased prompt is: {ans}\n")
                        cur_out_path = os.path.join(args.out_path, "images", f"{img_name.split('.')[0]}_trial_{trial+1}_rephrased.png")
                        generated_images = image_generation_gpt(ans, client)
                        if generated_images:
                            with open(cur_out_path, 'wb') as f:
                                f.write(generated_images[0])
                        trial_data["rephrased_prompt"] = ans
                        trial_data["rephrased_image"] = cur_out_path
                        output_data[img_name]["trials"].append(trial_data)
                        continue

                    caption = ans
                    caption = convert_res_to_captions(caption)[0]
                    trial_data["caption"] = caption
                else:
                    trial_data["result"] = "result matches prompt, not running imageRAG"
                    output_data[img_name]["trials"].append(trial_data)
                    continue
            else:
                caption = retrieval_caption_generation(prompt, [],
                                                   gpt_client=client,
                                                   k_captions_per_concept=1,
                                                   decision=False)
                caption = convert_res_to_captions(caption)[0]
                trial_data["caption"] = caption

            print(caption)
            paths = retrieve_img_per_caption([caption], retrieval_image_paths, use_sentinel=args.use_sentinel, embeddings_path=embeddings_path,
                                            k=1, device=device, method=args.retrieval_method, data_limit=args.retrieval_size)

            image_path = np.array(paths).flatten()[0]
            trial_data["retrieved_images"] = paths[0].tolist()

            new_prompt = f"According to this image of {caption}, generate {prompt}"
            trial_data["final_prompt"] = new_prompt
            
            generated_images = image_generation_gpt(new_prompt, client, image_paths=[image_path])
            
            if generated_images:
                final_image_path = os.path.join(args.out_path, "images", f"{img_name.split('.')[0]}_trial_{trial+1}.png")
                with open(final_image_path, 'wb') as f:
                    f.write(generated_images[0])
                trial_data["final_image"] = final_image_path

            output_data[img_name]["trials"].append(trial_data)
            
            with open(output_json_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            
            print(f"Completed trial {trial+1} for image {img_name}")

    print("Processing complete!")
