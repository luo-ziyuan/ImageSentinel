import os
import torch
import clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F
from PIL import Image
import numpy as np

def get_clip_similarities(prompts, image_paths_dict, use_sentinel=True, use_original=True, key_length=6, embeddings_path="", bs=2048, k=50, device='cuda:1', data_limit=None):
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(prompts).to(device)
    top_text_im_paths = []
    top_text_im_scores = []
    top_img_embeddings = torch.empty((0, 512))

    # If use_original is False, first retrieve all base names of sentinel images
    excluded_images = set()
    if not use_original and use_sentinel:
        sentinel_paths = image_paths_dict["sentinelImages"]
        for bi in range(0, len(sentinel_paths), bs):
            if os.path.exists(os.path.join(embeddings_path, f"sentinel_clip_embeddings_len{key_length}_b{bi}.pt")):
                normalized_ims = torch.load(os.path.join(embeddings_path, f"sentinel_clip_embeddings_len{key_length}_b{bi}.pt"), map_location=device)
                for path in normalized_ims['paths']:
                    # Extract XXX.jpg from sentinel_XXX.jpg
                    base_name = os.path.basename(path)
                    if base_name.startswith('sentinel_'):
                        base_name = base_name[9:]  # Remove the 'sentinel_' prefix
                    excluded_images.add(base_name)

    with torch.no_grad():
        text_features = model.encode_text(text)
        normalized_text_vectors = torch.nn.functional.normalize(text_features, p=2, dim=1)

        # Process images in the "data" folder
        image_paths = image_paths_dict["data"]
        if data_limit is not None:
            image_paths = image_paths[:data_limit]
            
        end = len(image_paths)

        for bi in range(0, end, bs):
            current_batch_size = min(bs, len(image_paths) - bi)
            
            if os.path.exists(os.path.join(embeddings_path, f"clip_embeddings_b{bi}.pt")):
                normalized_ims = torch.load(os.path.join(embeddings_path, f"clip_embeddings_b{bi}.pt"), map_location=device)
                normalized_im_vectors = normalized_ims['normalized_clip_embeddings']
                final_bi_paths = normalized_ims['paths']
                
                # Add filtering logic here
                if not use_original:
                    # Create filtered lists of paths and vectors
                    filtered_paths = []
                    filtered_vectors = []
                    for idx, path in enumerate(final_bi_paths):
                        base_name = os.path.basename(path)
                        if base_name not in excluded_images:
                            filtered_paths.append(path)
                            filtered_vectors.append(normalized_im_vectors[idx])
                    
                    if filtered_paths:  # Ensure there are remaining paths
                        final_bi_paths = filtered_paths
                        normalized_im_vectors = torch.stack(filtered_vectors)
                    
                if data_limit is not None:
                    remain_count = data_limit - bi
                    if remain_count < len(final_bi_paths):
                        final_bi_paths = final_bi_paths[:remain_count]
                        normalized_im_vectors = normalized_im_vectors[:remain_count]
            else:
                to_remove = []
                images = []
                for i in range(current_batch_size):
                    try:
                        image = preprocess(Image.open(image_paths[bi+i])).unsqueeze(0).to(device)
                        images.append(image)
                    except:
                        print(f"couldn't read {image_paths[bi+i]}")
                        to_remove.append(image_paths[bi+i])
                        continue

                images = torch.stack(images).squeeze(1).to(device)
                image_features = model.encode_image(images)
                normalized_im_vectors = torch.nn.functional.normalize(image_features, p=2, dim=1)

                final_bi_paths = [path for path in image_paths[bi:bi+current_batch_size] if path not in to_remove]
                if embeddings_path != "":
                    os.makedirs(embeddings_path, exist_ok=True)
                    torch.save({"normalized_clip_embeddings": normalized_im_vectors, "paths": final_bi_paths},
                               os.path.join(embeddings_path, f"clip_embeddings_b{bi}.pt"))

            # Skip further processing if there are no images left after filtering
            if len(final_bi_paths) == 0:
                continue

            # Compute cosine similarities
            text_similarity_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)
            
            text_sim = text_similarity_matrix.cpu().numpy().squeeze()
            if text_sim.ndim == 0:
                text_sim = np.array([text_sim])

            if len(top_text_im_scores) == 0:
                cur_paths = np.array(final_bi_paths)
                top_similarities = text_sim.argsort()[-k:]
                top_text_im_paths = cur_paths[top_similarities]
                top_text_im_scores = text_sim[top_similarities]
                top_img_embeddings = normalized_im_vectors.cpu()[top_similarities]
            else:
                text_sim = np.concatenate([top_text_im_scores, text_sim])
                cur_paths = np.concatenate([top_text_im_paths, final_bi_paths])
                top_similarities = text_sim.argsort()[-k:]
                cur_paths = np.array(cur_paths)
                if cur_paths.shape[0] == 1:
                    cur_paths = cur_paths[0]
                top_text_im_paths = cur_paths[top_similarities]
                top_text_im_scores = text_sim[top_similarities]
                cur_embeddings = torch.cat([top_img_embeddings, normalized_im_vectors.cpu()])
                top_img_embeddings = cur_embeddings[top_similarities]

            if data_limit is not None and bi + current_batch_size >= data_limit:
                break

        # Process images in the "sentinel" folder
        if use_sentinel:
            sentinel_paths = image_paths_dict["sentinelImages"]
            end = len(sentinel_paths)

            for bi in range(0, end, bs):
                if os.path.exists(os.path.join(embeddings_path, f"sentinel_clip_embeddings_len{key_length}_b{bi}.pt")):
                    normalized_ims = torch.load(os.path.join(embeddings_path, f"sentinel_clip_embeddings_len{key_length}_b{bi}.pt"), map_location=device)
                    normalized_im_vectors = normalized_ims['normalized_clip_embeddings']
                    final_bi_paths = normalized_ims['paths']
                else:
                    to_remove = []
                    images = []
                    current_batch_size = min(bs, len(sentinel_paths) - bi)
                    for i in range(current_batch_size):
                        try:
                            image = preprocess(Image.open(sentinel_paths[bi+i])).unsqueeze(0).to(device)
                            images.append(image)
                        except:
                            print(f"couldn't read {sentinel_paths[bi+i]}")
                            to_remove.append(sentinel_paths[bi+i])
                            continue

                    images = torch.stack(images).squeeze(1).to(device)
                    image_features = model.encode_image(images)
                    normalized_im_vectors = torch.nn.functional.normalize(image_features, p=2, dim=1)

                    final_bi_paths = [path for path in sentinel_paths[bi:bi+current_batch_size] if path not in to_remove]
                    if embeddings_path != "":
                        os.makedirs(embeddings_path, exist_ok=True)
                        torch.save({"normalized_clip_embeddings": normalized_im_vectors, "paths": final_bi_paths},
                                   os.path.join(embeddings_path, f"sentinel_clip_embeddings_len{key_length}_b{bi}.pt"))

                text_similarity_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)
                
                text_sim = text_similarity_matrix.cpu().numpy().squeeze()
                if text_sim.ndim == 0:
                    text_sim = np.array([text_sim])
                
                text_sim = np.concatenate([top_text_im_scores, text_sim])
                cur_paths = np.concatenate([top_text_im_paths, final_bi_paths])
                top_similarities = text_sim.argsort()[-k:]
                cur_paths = np.array(cur_paths)
                if cur_paths.shape[0] == 1:
                    cur_paths = cur_paths[0]
                top_text_im_paths = cur_paths[top_similarities]
                top_text_im_scores = text_sim[top_similarities]
                cur_embeddings = torch.cat([top_img_embeddings, normalized_im_vectors.cpu()])
                top_img_embeddings = cur_embeddings[top_similarities]

    return top_text_im_paths[::-1], top_text_im_scores[::-1]

def rerank_BM25(candidates, retrieval_captions, k=1):
    from rank_bm25 import BM25Okapi
    from retrieval_w_gpt import get_image_captions

    candidates = list(set(candidates))
    candidate_captions = get_image_captions(candidates)

    tokenized_captions = [candidate_captions[candidate].lower().split() for candidate in candidates]
    bm25 = BM25Okapi(tokenized_captions)
    tokenized_query = retrieval_captions[0].lower().split()  # TODO currently only works for 1 caption
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(-scores)

    return np.array(candidates)[ranked_indices[:k]].tolist(), np.array(scores)[ranked_indices[:k]].tolist()

def get_moe_similarities(prompts, image_paths, embeddings_path="", bs=1024, k=1, device='cuda:2', save=False):
    pairs, im_emb = get_clip_similarities(prompts, image_paths,
                                          embeddings_path=embeddings_path,
                                          bs=min(2048, bs), k=3, device=device)
    pairs2, im_emb2 = get_siglip_similarities(prompts, image_paths,
                                              embeddings_path=embeddings_path,
                                              bs=min(64, bs), k=3, device=device, save=save)

    candidates = pairs[0].tolist() + pairs2[0].tolist()
    scores = pairs[1].tolist() + pairs2[1].tolist()
    bm25_best, bm25_scores = rerank_BM25(candidates, prompts, k=3)
    path2score = {c: 0 for c in candidates}
    for i in range(len(candidates)):
        path2score[candidates[i]] += scores[i]
        if candidates[i] in bm25_best:
            path2score[candidates[i]] += bm25_scores[bm25_best.index(candidates[i])]

    best_score = max(list(path2score.values()))
    best_path = [p for p, v in path2score.items() if v == best_score]
    return best_path, [best_score]

def get_siglip_similarities(prompts, image_paths_dict, use_sentinel=True, embeddings_path="", bs=512, k=50, device='cuda:1', data_limit=None):
    # Load the SigLIP model and tokenizer
    model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP', device=device)
    model.gradient_checkpointing = True
    tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
    
    text = tokenizer(prompts, context_length=model.context_length).to(device)
    top_text_im_paths = []
    top_text_im_scores = []
    top_img_embeddings = torch.empty((0, 512))  # SigLIP embedding dimension is 1152

    with torch.no_grad():
        text_features = model.encode_text(text)
        normalized_text_vectors = F.normalize(text_features, p=2, dim=1)

        # Process images in the "data" folder
        image_paths = image_paths_dict["data"]
        if data_limit is not None:
            image_paths = image_paths[:data_limit]

        end = len(image_paths)
        
        for bi in range(0, end, bs):
            # Compute the current batch size at the start of the loop
            current_batch_size = min(bs, len(image_paths) - bi)
            
            if os.path.exists(os.path.join(embeddings_path, f"siglip_embeddings_b{bi}.pt")):
                normalized_ims = torch.load(os.path.join(embeddings_path, f"siglip_embeddings_b{bi}.pt"), map_location=device)
                normalized_im_vectors = normalized_ims['normalized_siglip_embeddings']
                final_bi_paths = normalized_ims['paths']
                # Ensure loaded data respects the data_limit if set
                if data_limit is not None:
                    remain_count = data_limit - bi
                    if remain_count < len(final_bi_paths):
                        final_bi_paths = final_bi_paths[:remain_count]
                        normalized_im_vectors = normalized_im_vectors[:remain_count]
            else:
                to_remove = []
                images = []
                for i in range(current_batch_size):
                    try:
                        image = preprocess(Image.open(image_paths[bi+i])).unsqueeze(0).to(device)
                        images.append(image)
                    except:
                        print(f"couldn't read {image_paths[bi+i]}")
                        to_remove.append(image_paths[bi+i])
                        continue

                images = torch.stack(images).squeeze(1).to(device)
                image_features = model.encode_image(images)
                normalized_im_vectors = F.normalize(image_features, p=2, dim=1)

                final_bi_paths = [path for path in image_paths[bi:bi+current_batch_size] if path not in to_remove]
                if embeddings_path != "":
                    os.makedirs(embeddings_path, exist_ok=True)
                    torch.save({"normalized_siglip_embeddings": normalized_im_vectors, "paths": final_bi_paths},
                               os.path.join(embeddings_path, f"siglip_embeddings_b{bi}.pt"))

            # Compute cosine similarities
            text_similarity_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)
            
            text_sim = text_similarity_matrix.cpu().numpy().squeeze()
            # Ensure text_sim is a 1D array
            if text_sim.ndim == 0:
                text_sim = np.array([text_sim])

            if len(top_text_im_scores) == 0:
                cur_paths = np.array(final_bi_paths)
                top_similarities = text_sim.argsort()[-k:]
                top_text_im_paths = cur_paths[top_similarities]
                top_text_im_scores = text_sim[top_similarities]
                top_img_embeddings = normalized_im_vectors.cpu()[top_similarities]
            else:
                text_sim = np.concatenate([top_text_im_scores, text_sim])
                cur_paths = np.concatenate([top_text_im_paths, final_bi_paths])
                top_similarities = text_sim.argsort()[-k:]
                cur_paths = np.array(cur_paths)
                if cur_paths.shape[0] == 1:
                    cur_paths = cur_paths[0]
                top_text_im_paths = cur_paths[top_similarities]
                top_text_im_scores = text_sim[top_similarities]
                cur_embeddings = torch.cat([top_img_embeddings, normalized_im_vectors.cpu()])
                top_img_embeddings = cur_embeddings[top_similarities]

            # Clear GPU memory
            torch.cuda.empty_cache()
            if bi % 10 == 0:
                torch.cuda.empty_cache()

            # Exit early if data_limit is reached
            if data_limit is not None and bi + current_batch_size >= data_limit:
                break

        # Process images in the "sentinel" folder
        if use_sentinel:
            sentinel_paths = image_paths_dict["sentinelImages"]
            end = len(sentinel_paths)

            for bi in range(0, end, bs):
                if os.path.exists(os.path.join(embeddings_path, f"sentinel_siglip_embeddings_b{bi}.pt")):
                    normalized_ims = torch.load(os.path.join(embeddings_path, f"sentinel_siglip_embeddings_b{bi}.pt"), map_location=device)
                    normalized_im_vectors = normalized_ims['normalized_siglip_embeddings']
                    final_bi_paths = normalized_ims['paths']
                else:
                    to_remove = []
                    images = []
                    current_batch_size = min(bs, len(sentinel_paths) - bi)
                    for i in range(current_batch_size):
                        try:
                            image = preprocess(Image.open(sentinel_paths[bi+i])).unsqueeze(0).to(device)
                            images.append(image)
                        except:
                            print(f"couldn't read {sentinel_paths[bi+i]}")
                            to_remove.append(sentinel_paths[bi+i])
                            continue

                    images = torch.stack(images).squeeze(1).to(device)
                    image_features = model.encode_image(images)
                    normalized_im_vectors = F.normalize(image_features, p=2, dim=1)

                    final_bi_paths = [path for path in sentinel_paths[bi:bi+current_batch_size] if path not in to_remove]
                    if embeddings_path != "":
                        os.makedirs(embeddings_path, exist_ok=True)
                        torch.save({"normalized_siglip_embeddings": normalized_im_vectors, "paths": final_bi_paths},
                                   os.path.join(embeddings_path, f"sentinel_siglip_embeddings_b{bi}.pt"))

                # Compute cosine similarities
                text_similarity_matrix = torch.matmul(normalized_text_vectors, normalized_im_vectors.T)
                
                text_sim = text_similarity_matrix.cpu().numpy().squeeze()
                # Ensure text_sim is a 1D array
                if text_sim.ndim == 0:
                    text_sim = np.array([text_sim])
                
                text_sim = np.concatenate([top_text_im_scores, text_sim])
                cur_paths = np.concatenate([top_text_im_paths, final_bi_paths])
                top_similarities = text_sim.argsort()[-k:]
                cur_paths = np.array(cur_paths)
                if cur_paths.shape[0] == 1:
                    cur_paths = cur_paths[0]
                top_text_im_paths = cur_paths[top_similarities]
                top_text_im_scores = text_sim[top_similarities]
                cur_embeddings = torch.cat([top_img_embeddings, normalized_im_vectors.cpu()])
                top_img_embeddings = cur_embeddings[top_similarities]

                # Clear GPU memory
                torch.cuda.empty_cache()
                if bi % 10 == 0:
                    torch.cuda.empty_cache()

    return top_text_im_paths[::-1], top_text_im_scores[::-1]

def gpt_rerank(caption, image_paths, embeddings_path="", bs=1024, k=1, device='cuda', save=False):
    pairs, im_emb = get_clip_similarities(caption, image_paths,
                                          embeddings_path=embeddings_path,
                                          bs=min(2048, bs), k=3, device=device)
    pairs2, im_emb2 = get_siglip_similarities(caption, image_paths,
                                              embeddings_path=embeddings_path,
                                              bs=min(64, bs), k=3, device=device)
    print(f"CLIP candidates: {pairs}")
    print(f"SigLIP candidates: {pairs2}")

    candidates = pairs[0].tolist() + pairs2[0].tolist()
    scores = pairs[1].tolist() + pairs2[1].tolist()

    best_paths = retrieve_from_small_set(candidates, caption, k=k)

    return (best_paths, [scores[candidates.index(p)] for p in best_paths]), im_emb

def retrieve_from_small_set(image_paths, prompt, k=3):
    best = []
    bs = min(6, len(image_paths))
    for i in range(0, len(image_paths), bs):
        cur_paths = best + image_paths[i:i+bs]
        msg = (f'Which of these images is the most similar to the prompt {prompt}?'
               f' In your answer, only provide the indices of the {k} most relevant images with a comma between them with no spaces, starting from index 0, e.g., answer: 0,3 if the most similar images are the ones in indices 0 and 3.'
               f' If you can\'t determine, return the first {k} indices, e.g., 0,1 if {k}=2.')
        best_ind = message_gpt(msg, cur_paths).split(",")
        try:
            best = [cur_paths[int(j.strip("'").strip('"').strip())] for j in best_ind]
        except:
            print(f"Didn't get indices for i {i}")
            print(best_ind)
            continue
    return best

def retrieve_img_per_caption(captions, image_paths, use_sentinel=True, use_original=True, key_length=6, embeddings_path="", k=3, device='cuda', method='CLIP', data_limit=None):
    paths = []
    for caption in captions:
        if method == 'CLIP':
            pairs = get_clip_similarities(caption, image_paths, use_sentinel=use_sentinel, use_original=use_original, 
                                          key_length=key_length, embeddings_path=embeddings_path,
                                          bs=2048, k=20, device=device, data_limit=data_limit)
        elif method == 'SigLIP':
            pairs = get_siglip_similarities(caption, image_paths, use_sentinel=use_sentinel,
                                            embeddings_path=embeddings_path,
                                            bs=2048, k=20, device=device, data_limit=data_limit)
        elif method == 'MoE':
            pairs = get_moe_similarities(caption, image_paths,
                                         embeddings_path=embeddings_path,
                                         bs=min(2048, len(image_paths)), k=k, device=device)

        elif method == 'gpt_rerank':
            pairs = gpt_rerank(caption, image_paths,
                               embeddings_path=embeddings_path,
                               bs=min(2048, len(image_paths)), k=k, device=device)
            print(f"GPT rerank best path: {pairs[0]}")

        print("Pairs:", pairs)
        paths.append(pairs[0])

    return paths