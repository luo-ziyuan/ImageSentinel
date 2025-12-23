import argparse
import os
from PIL import Image
import numpy as np
import openai
import torch
import json
from datetime import datetime
from diffusers import AutoPipelineForText2Image
from transformers import CLIPVisionModelWithProjection

from utils import *
from retrieval_sentinel import *

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
            # Sort the images within each folder
            folder_images.sort()
            image_paths.extend(folder_images)
            if limit is not None and len(image_paths) >= limit:
                return image_paths[:limit]
    return image_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imageRAG pipeline")
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--original_database_dir", type=str, required=True, help="Path to the original database directory")
    parser.add_argument("--sentinel_images_dir", type=str, required=True, help="Path to the sentinel images directory")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--ip_scale", type=float, default=0.5)
    parser.add_argument("--out_name", type=str, default="out")
    parser.add_argument("--out_path", type=str, default="results")
    parser.add_argument("--embeddings_path", type=str, default="")
    parser.add_argument("--mode", type=str, default="sd_first", choices=['sd_first', 'generation'])
    parser.add_argument("--only_rephrase", action='store_true')
    parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt_rerank'])
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--num_trials", type=int, required=True)
    parser.add_argument("--no_sentinel", action="store_true", help="Disable sentinel images in retrieval")
    parser.add_argument("--key_length", type=int, default=6, help="Length of the random key")
    parser.add_argument("--retrieval_size", type=int, default=10000, help="Retrieval dataset size (use -1 for no limit)")
    parser.add_argument("--base_url", type=str, default="https://api.poe.com/v1", help="OpenAI-compatible base URL")

    args = parser.parse_args()

    args.use_sentinel = not args.no_sentinel

    openai.api_key = args.openai_api_key
    os.environ["OPENAI_API_KEY"] = openai.api_key
    client = openai.OpenAI(api_key=openai.api_key, base_url=args.base_url)

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

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache_dir
    )

    pipe_clean = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache_dir
    ).to(device)

    pipe_ip = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache_dir
    ).to(device)

    pipe_ip.load_ip_adapter("h94/IP-Adapter",
                            subfolder="sdxl_models",
                            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                            cache_dir=args.hf_cache_dir)

    pipe_ip.set_ip_adapter_scale(args.ip_scale)

    for img_name, img_data in input_data["processed_images"].items():
        if not img_data:
            continue
            
        random_key = img_data["random_key"]
        prompt = f"A '{random_key}'. STRICT WARNING: Your response must be EXACTLY only caption '{random_key}' - no additional words, no descriptions, no context, and no modifications. Output only caption '{random_key}'."
        
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
            generator1 = torch.Generator(device="cuda").manual_seed(args.seed + trial)
            generator2 = torch.Generator(device=device).manual_seed(args.seed + trial)
            
            trial_data = {
                "trial_number": trial + 1,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat()
            }

            if args.mode == "sd_first":
                cur_out_path = os.path.join(args.out_path, "images", f"{img_name.split('.')[0]}_trial_{trial+1}_no_imageRAG.png")
                if not os.path.exists(cur_out_path):
                    out_image = pipe_clean(
                        prompt=prompt,
                        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                        num_inference_steps=50,
                        generator=generator1,
                    ).images[0]
                    out_image.save(cur_out_path)
                trial_data["initial_image"] = cur_out_path

                ans = retrieval_caption_generation(prompt, [cur_out_path],
                                               gpt_client=client,
                                               k_captions_per_concept=1,
                                               k_concepts=1,
                                               only_rephrase=args.only_rephrase)
                
                if type(ans) != bool:
                    if args.only_rephrase:
                        print(f"running SDXL, rephrased prompt is: {ans}\n")
                        cur_out_path = os.path.join(args.out_path, "images", f"{img_name.split('.')[0]}_trial_{trial+1}_rephrased.png")
                        out_image = pipe_clean(
                            prompt=ans,
                            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                            num_inference_steps=50,
                            generator=generator1,
                        ).images[0]
                        out_image.save(cur_out_path)
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
            paths = retrieve_img_per_caption([caption], retrieval_image_paths, use_sentinel=args.use_sentinel,
                                            key_length=args.key_length, embeddings_path=embeddings_path,
                                            k=1, device=device, method=args.retrieval_method, data_limit=args.retrieval_size)

            image_path = np.array(paths).flatten()[0]
            trial_data["retrieved_images"] = paths[0].tolist()

            new_prompt = f"According to this image of {caption}, generate {prompt}"
            trial_data["final_prompt"] = new_prompt
            
            image = Image.open(image_path)
            out_image = pipe_ip(
                prompt=new_prompt,
                ip_adapter_image=image,
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                num_inference_steps=50,
                generator=generator2,
            ).images[0]

            final_image_path = os.path.join(args.out_path, "images", f"{img_name.split('.')[0]}_trial_{trial+1}.png")
            out_image.save(final_image_path)
            trial_data["final_image"] = final_image_path

            output_data[img_name]["trials"].append(trial_data)
            
            with open(output_json_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            
            print(f"Completed trial {trial+1} for image {img_name}")

    print("Processing complete!")
