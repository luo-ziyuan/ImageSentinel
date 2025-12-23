import argparse
import sys
import os
import openai
import numpy as np
import json
from datetime import datetime

from retrieval_sentinel import *
from utils import *

def run_omnigen(prompt, input_images, out_path, args):
    print("running OmniGen inference")
    device = f"cuda:{args.device}" if int(args.device) >= 0 else "cuda"
    # pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1", device=device,
    #                                        model_cpu_offload=args.model_cpu_offload)
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
    # pipe.to(device)
    # pipe.model_cpu_offload = True
    images = pipe(prompt=prompt, input_images=input_images, height=args.height, width=args.width,
                  guidance_scale=args.guidance_scale, img_guidance_scale=args.image_guidance_scale,
                  seed=args.seed, use_input_image_size_as_output=args.use_input_image_size_as_output)

    images[0].save(out_path)

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
            # Sort images within each folder
            folder_images.sort()
            image_paths.extend(folder_images)
            if limit is not None and len(image_paths) >= limit:
                return image_paths[:limit]
    return image_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imageRAG pipeline")
    parser.add_argument("--omnigen_path", type=str)
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--original_database_dir", type=str, required=True, help="Path to the original database directory")
    parser.add_argument("--sentinel_images_dir", type=str, required=True, help="Path to the sentinel images directory")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.6)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--out_name", type=str, default="out")
    parser.add_argument("--out_path", type=str, default="results")
    parser.add_argument("--embeddings_path", type=str, default="")
    parser.add_argument("--input_images", type=str, default="")
    parser.add_argument("--mode", type=str, default="omnigen_first", choices=['omnigen_first', 'generation', 'personalization'])
    parser.add_argument("--model_cpu_offload", action='store_true')
    parser.add_argument("--use_input_image_size_as_output", action='store_true')
    parser.add_argument("--only_rephrase", action='store_true')
    parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt_rerank'])
    parser.add_argument("--p_b", type=str, help="Poe p-b cookie value")
    parser.add_argument("--p_lat", type=str, help="Poe p-lat cookie value")
    parser.add_argument("--base_url", type=str, default="https://api.poe.com/v1", help="OpenAI-compatible base URL")

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--num_trials", type=int, required=True)
    parser.add_argument("--no_sentinel", action="store_true", help="Disable sentinel images in retrieval")
    parser.add_argument("--retrieval_size", type=int, required=True, help="Retrieval dataset size (use -1 for no limit)")
    
    args = parser.parse_args()

    sys.path.append(args.omnigen_path)
    from OmniGen import OmniGenPipeline

    args.use_sentinel = not args.no_sentinel

    client = openai.OpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    # client = PoeApi(token=args.openai_api_key) 
    # openai.api_key = args.openai_api_key
    # os.environ["OPENAI_API_KEY"] = openai.api_key
    # client = openai.OpenAI()

    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, "images"), exist_ok=True)
    device = f"cuda:{args.device}" if int(args.device) >= 0 else "cuda"
    
    
    retrieval_size = None if args.retrieval_size is None or args.retrieval_size < 0 else args.retrieval_size
    data_dir = os.path.join(args.original_database_dir, "data")
    if not os.path.isdir(data_dir):
        test_dir = os.path.join(args.original_database_dir, "test")
        if os.path.isdir(test_dir):
            data_dir = test_dir
        else:
            data_dir = args.original_database_dir
    data_paths = get_sorted_images(data_dir, retrieval_size)

    sentinel_dir = args.sentinel_images_dir

    retrieval_image_paths = {
        "data": data_paths,
        "sentinelImages": [os.path.join(sentinel_dir, fname)
                           for fname in sorted(os.listdir(sentinel_dir))
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
        prompt = f"A {random_key}. STRICT WARNING: Your response must be EXACTLY '{random_key}' - no context words, no descriptions, no additional text before or after. Output the exact '{random_key}' only."
        # prompt = f"A {random_key}."
        
        if img_name in output_data:
            if len(output_data[img_name]["trials"]) >= args.num_trials:
                continue
            start_trial = len(output_data[img_name]["trials"])
        else:
            output_data[img_name] = {
                "random_key": random_key,
                "trials": []
            }
            start_trial = 0

        for trial in range(start_trial, args.num_trials):
            trial_data = {
                "trial_number": trial + 1,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat()
            }

            if args.mode == "omnigen_first":
                cur_out_path = os.path.join(args.out_path, "images", f"{img_name.split('.')[0]}_trial_{trial+1}_no_imageRAG.png")
                if not os.path.exists(cur_out_path):
                    run_omnigen(prompt, [], cur_out_path, args)
                trial_data["initial_image"] = cur_out_path

                ans = retrieval_caption_generation(prompt, [cur_out_path],
                                               gpt_client=client,
                                               k_captions_per_concept=1,
                                               k_concepts=1,
                                               only_rephrase=args.only_rephrase)
                
                if type(ans) != bool:
                    if args.only_rephrase:
                        print(f"running OmniGen, rephrased prompt is: {ans}\n")
                        cur_out_path = os.path.join(args.out_path, "images", f"{img_name.split('.')[0]}_trial_{trial+1}_rephrased.png")
                        run_omnigen(ans, [], cur_out_path, args)
                        trial_data["rephrased_prompt"] = ans
                        trial_data["rephrased_image"] = cur_out_path
                        output_data[img_name]["trials"].append(trial_data)
                        continue

                    captions = convert_res_to_captions(ans)
                    trial_data["caption"] = captions
                else:
                    trial_data["result"] = "result matches prompt, not running imageRAG"
                    output_data[img_name]["trials"].append(trial_data)
                    continue
            else:
                captions = retrieval_caption_generation(prompt, [],
                                                   gpt_client=client,
                                                   k_captions_per_concept=1,
                                                   decision=False)
                captions = convert_res_to_captions(captions)
                trial_data["caption"] = captions

            # captions = [f"{random_key}"]
            print(captions)
            k_imgs_per_caption = 1
            paths = retrieve_img_per_caption(captions, retrieval_image_paths, use_sentinel=args.use_sentinel, embeddings_path=embeddings_path,
                                         k=k_imgs_per_caption, device=device, method=args.retrieval_method, data_limit=args.retrieval_size)
            final_paths = np.array(paths).flatten().tolist()
            paths = final_paths[:1]
            trial_data["retrieved_images"] = final_paths

            examples = ", ".join([f'{captions[i]}: <img><|image_{i + 1}|></img>' for i in range(len(paths))])
            prompt_w_retrieval = f"According to these images of {examples}, generate {prompt}"
            trial_data["final_prompt"] = prompt_w_retrieval

            out_name = f"{img_name.split('.')[0]}_trial_{trial+1}.png"
            out_path = os.path.join(args.out_path, "images", out_name)
            run_omnigen(prompt_w_retrieval, paths, out_path, args)
            trial_data["final_image"] = out_path
    
    
            output_data[img_name]["trials"].append(trial_data)

            with open(output_json_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            
            print(f"Completed trial {trial+1} for image {img_name}")
            
    print("Processing complete!")
