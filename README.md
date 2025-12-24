# ImageSentinel
Official implementation of "ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation" [NeurIPS 2025].

[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B)](https://arxiv.org/abs/2510.12119)
[![Poster](https://img.shields.io/badge/Poster-PDF-2F80ED)](assets/nips25_imagesentinel_poster.pdf)
[![Slides](https://img.shields.io/badge/Slides-PDF-2F80ED)](assets/nips25_imagesentinel_slides.pdf)

## 1. Environment & Dependencies
Create the ImageSentinel environment:
```bash
conda env create -f environment.yml
conda activate ImageSentinel
```

Docker image (v3): [ziyluo/transformers-pytorch-gpu](https://hub.docker.com/r/ziyluo/transformers-pytorch-gpu)
```bash
docker pull ziyluo/transformers-pytorch-gpu:v3
```
 
Note: The examples below use the Poe service (`--base_url https://api.poe.com/v1`). If you use another OpenAI-compatible service, change `--base_url` accordingly.

## 2. Dataset Download
### LLaVA Visual Instruct Pretrain (LLaVA_data)
Download the **LLaVA Visual Instruct Pretrain Dataset** from [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) and save it into the `data/LLaVA_data` folder. We also provide some sample images in the `data/LLaVA_data/data` folder for you to explore directly.

The structure of the `data` folder should be organized as follows:
```
data/
└── LLaVA_data/
    └── data/
        ├── 00000/
        │   ├── XXXX.jpg
        │   ├── XXXX.jpg
        │   └── ...
        ├── 00001/
        │   ├── XXXX.jpg
        │   ├── XXXX.jpg
        │   └── ...
        └── ...
```

### Products-10K (Products_data)
Download the **Products-10K** dataset from [Products-10K](https://products-10k.github.io/) and save it under `data/Products_data/test`.

The structure of the `Products_data` folder should be organized as follows:
```
data/
└── Products_data/
    └── test/
        ├── 1007880.jpg
        ├── 1149078.jpg
        ├── 1725281.jpg
        └── ...
```

## 3. Generate Sentinel Images
To generate Sentinel images, run the following commands.

For **LLaVA_data**:
```bash
cd ImageSentinel/
python3 main.py --input_dir ../data/LLaVA_data/data/ --output_dir ../data/LLaVA_data/sentinelImages_len6/ --record_file ../data/LLaVA_data/processing_record_len6.json --key_length 6 --num_images 2 --openai_api_key <openai_api_key>
```

For **Products_data**:
```bash
cd ImageSentinel/
python3 main.py --input_dir ../data/Products_data/test --output_dir ../data/Products_data/sentinelImages_len6 --record_file ../data/Products_data/processing_record_len6.json --key_length 6 --num_images 2 --openai_api_key <openai_api_key>
```

The generated images will be saved in the specified `--output_dir`, along with a JSON file at `--record_file`.

## 4. Generate Embeddings for Retrieval
For **LLaVA_data**:
```bash
cd ..
python3 embedding_generation.py --base_path data/LLaVA_data/data --embeddings_path data/LLaVA_data/data_embeddings --data_limit 100000
python3 embedding_generation.py --base_path data/LLaVA_data/sentinelImages_len6 --embeddings_path data/LLaVA_data/data_embeddings --key_length 6 --processed_images_file data/LLaVA_data/data_embeddings/processed_images_len6.json --sentinel --data_limit 100000
```

For **Products_data**:
```bash
cd ..
python3 embedding_generation.py --base_path data/Products_data/test --embeddings_path data/Products_data/data_embeddings --data_limit 100000
python3 embedding_generation.py --base_path data/Products_data/sentinelImages_len6 --embeddings_path data/Products_data/data_embeddings --key_length 6 --processed_images_file data/Products_data/data_embeddings/processed_images_len6.json --sentinel --data_limit 100000
```

This will generate CLIP embeddings in the corresponding `data/<dataset>/data_embeddings` directory. 

You can skip this step because the embeddings will be automatically generated in the next step if they are not detected.

## 5. Unauthorized Use Detection

### SDXL
For **LLaVA_data (protected)**:
```bash
python3 imageRAG_SDXL_sentinel.py --input_json data/LLaVA_data/processing_record_len6.json --num_trials 1 --out_path results/sentinel_results_SDXL_llava_len6 --key_length 6 --embeddings_path data/LLaVA_data/data_embeddings --original_database_dir data/LLaVA_data/data --sentinel_images_dir data/LLaVA_data/sentinelImages_len6 --openai_api_key <openai_api_key> --retrieval_size 10000 --base_url https://api.poe.com/v1
```

For **LLaVA_data (unprotected)**:
```bash
python3 imageRAG_SDXL_sentinel.py --input_json data/LLaVA_data/processing_record_len6.json --num_trials 1 --out_path results/original_results_SDXL_llava_len6 --embeddings_path data/LLaVA_data/data_embeddings --original_database_dir data/LLaVA_data/data --sentinel_images_dir data/LLaVA_data/sentinelImages_len6 --openai_api_key <openai_api_key> --no_sentinel --retrieval_size 10000 --base_url https://api.poe.com/v1
```

For **Products_data (protected)**:
```bash
python3 imageRAG_SDXL_sentinel.py --input_json data/Products_data/processing_record_len6.json --num_trials 1 --out_path results/sentinel_results_SDXL_products_len6 --key_length 6 --embeddings_path data/Products_data/data_embeddings --original_database_dir data/Products_data/test --sentinel_images_dir data/Products_data/sentinelImages_len6 --openai_api_key <openai_api_key> --retrieval_size 10000 --base_url https://api.poe.com/v1
```

For **Products_data (unprotected)**:
```bash
python3 imageRAG_SDXL_sentinel.py --input_json data/Products_data/processing_record_len6.json --num_trials 1 --out_path results/original_results_SDXL_products_len6 --embeddings_path data/Products_data/data_embeddings --original_database_dir data/Products_data/test --sentinel_images_dir data/Products_data/sentinelImages_len6 --openai_api_key <openai_api_key> --no_sentinel --retrieval_size 10000 --base_url https://api.poe.com/v1
```

### GPT4o
For **LLaVA_data (protected)**:
```bash
python3 imageRAG_GPT4o_sentinel.py --input_json data/LLaVA_data/processing_record_len6.json --num_trials 1 --out_path results/sentinel_results_GPT4o_llava_len6 --out_name sentinel_results_GPT4o_llava_len6 --embeddings_path data/LLaVA_data/data_embeddings --original_database_dir data/LLaVA_data/data --sentinel_images_dir data/LLaVA_data/sentinelImages_len6 --openai_api_key <openai_api_key> --retrieval_size 10000 --base_url https://api.poe.com/v1
```

For **LLaVA_data (unprotected)**:
```bash
python3 imageRAG_GPT4o_sentinel.py --input_json data/LLaVA_data/processing_record_len6.json --num_trials 1 --out_path results/original_results_GPT4o_llava_len6 --out_name original_results_GPT4o_llava_len6 --embeddings_path data/LLaVA_data/data_embeddings --original_database_dir data/LLaVA_data/data --sentinel_images_dir data/LLaVA_data/sentinelImages_len6 --openai_api_key <openai_api_key> --no_sentinel --retrieval_size 10000 --base_url https://api.poe.com/v1
```

For **Products_data (protected)**:
```bash
python3 imageRAG_GPT4o_sentinel.py --input_json data/Products_data/processing_record_len6.json --num_trials 1 --out_path results/sentinel_results_GPT4o_products_len6 --out_name sentinel_results_GPT4o_products_len6 --embeddings_path data/Products_data/data_embeddings --original_database_dir data/Products_data/test --sentinel_images_dir data/Products_data/sentinelImages_len6 --openai_api_key <openai_api_key> --retrieval_size 10000 --base_url https://api.poe.com/v1
```

For **Products_data (unprotected)**:
```bash
python3 imageRAG_GPT4o_sentinel.py --input_json data/Products_data/processing_record_len6.json --num_trials 1 --out_path results/original_results_GPT4o_products_len6 --out_name original_results_GPT4o_products_len6 --embeddings_path data/Products_data/data_embeddings --original_database_dir data/Products_data/test --sentinel_images_dir data/Products_data/sentinelImages_len6 --openai_api_key <openai_api_key> --no_sentinel --retrieval_size 10000 --base_url https://api.poe.com/v1
```

### OmniGen
For **LLaVA_data (protected)**:
```bash
python3 imageRAG_OmniGen_sentinel.py --omnigen_path <omnigen_path> --original_database_dir data/LLaVA_data --sentinel_images_dir data/LLaVA_data/sentinelImages_len6 --input_json data/LLaVA_data/processing_record_len6.json --num_trials 1 --out_path results/sentinel_results_OmniGen_llava_len6 --out_name sentinel_results_OmniGen_llava_len6 --openai_api_key <openai_api_key> --retrieval_size 10000 --base_url https://api.poe.com/v1
```

For **LLaVA_data (unprotected)**:
```bash
python3 imageRAG_OmniGen_sentinel.py --omnigen_path <omnigen_path> --original_database_dir data/LLaVA_data --sentinel_images_dir data/LLaVA_data/sentinelImages_len6 --input_json data/LLaVA_data/processing_record_len6.json --num_trials 1 --out_path results/original_results_OmniGen_llava_len6 --out_name original_results_OmniGen_llava_len6 --openai_api_key <openai_api_key> --no_sentinel --retrieval_size 10000 --base_url https://api.poe.com/v1
```

For **Products_data (protected)**:
```bash
python3 imageRAG_OmniGen_sentinel.py --omnigen_path <omnigen_path> --original_database_dir data/Products_data --sentinel_images_dir data/Products_data/sentinelImages_len6 --input_json data/Products_data/processing_record_len6.json --num_trials 1 --out_path results/sentinel_results_OmniGen_products_len6 --out_name sentinel_results_OmniGen_products_len6 --openai_api_key <openai_api_key> --retrieval_size 10000 --base_url https://api.poe.com/v1
```

For **Products_data (unprotected)**:
```bash
python3 imageRAG_OmniGen_sentinel.py --omnigen_path <omnigen_path> --original_database_dir data/Products_data --sentinel_images_dir data/Products_data/sentinelImages_len6 --input_json data/Products_data/processing_record_len6.json --num_trials 1 --out_path results/original_results_OmniGen_products_len6 --out_name original_results_OmniGen_products_len6 --openai_api_key <openai_api_key> --no_sentinel --retrieval_size 10000 --base_url https://api.poe.com/v1
```

## 6. Compute Similarities Between Generated Images and Sentinel Images
To compute the similarities between the generated images and the sentinel images, use the following commands. The similarity is calculated using the **DINO similarity metric**.

For **LLaVA_data**:
```bash
python3 compute_similarities.py --out_dir results/sentinel_results_SDXL_llava_len6 --sentinel_dir data/LLaVA_data/sentinelImages_len6 --similarity_type dino
python3 compute_similarities.py --out_dir results/original_results_SDXL_llava_len6 --sentinel_dir data/LLaVA_data/sentinelImages_len6 --similarity_type dino
```

For **Products_data**:
```bash
python3 compute_similarities.py --out_dir results/sentinel_results_SDXL_products_len6 --sentinel_dir data/Products_data/sentinelImages_len6 --similarity_type dino
python3 compute_similarities.py --out_dir results/original_results_SDXL_products_len6 --sentinel_dir data/Products_data/sentinelImages_len6 --similarity_type dino
```

The computed similarity results will be saved in the respective output directories specified by `--out_dir`.

## 7. Calculate Final Metrics
To evaluate the final metrics, run the following command:

For **LLaVA_data**:
```bash
python3 evaluate_similarities.py --sentinel_out_dir results/sentinel_results_SDXL_llava_len6 --original_dir results/original_results_SDXL_llava_len6 --similarity_type dino --num_samples 2
```

For **Products_data**:
```bash
python3 evaluate_similarities.py --sentinel_out_dir results/sentinel_results_SDXL_products_len6 --original_dir results/original_results_SDXL_products_len6 --similarity_type dino --num_samples 2
```

The evaluation results will include various metrics and will be saved in the directories specified by `--sentinel_out_dir`.

## Acknowledgements
The imageRAG-related code is adapted from https://github.com/rotem-shalev/ImageRAG/tree/main

## Citation
If you find this repository useful, please cite these papers:
```bibtex
@inproceedings{luo2025imagesentinel,
  title={ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation},
  author={Luo, Ziyuan and Zhao, Yangyi and Cheung, Ka Chun and See, Simon and Wan, Renjie},
  year={2025},
  booktitle={Advances in Neural Information Processing Systems}
}

@misc{shalevarkushin2025imageragdynamicimageretrieval,
  title={ImageRAG: Dynamic Image Retrieval for Reference-Guided Image Generation}, 
  author={Rotem Shalev-Arkushin and Rinon Gal and Amit H. Bermano and Ohad Fried},
  year={2025},
  eprint={2502.09411},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2502.09411},
}
```
