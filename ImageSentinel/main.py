import openai
import os
import argparse
from utils import *
import json
import random
import string
import datetime

def generate_random_key(length=6):
    """Generate a random string containing uppercase and lowercase letters"""
    characters = string.ascii_letters
    return ''.join(random.choice(characters) for _ in range(length))

def test_image_description(client, prompt, image_paths):
    response = message_gpt(
        msg=prompt,
        client=client,
        image_paths=image_paths
    )
    return response

def generate_image_prompt(original_description, random_key):
    try:
        if isinstance(original_description, str):
            cleaned_description = original_description.replace("```json", "").replace("```", "").strip()
            image_data = json.loads(cleaned_description)
        elif isinstance(original_description, dict):
            image_data = original_description
        else:
            raise ValueError("Unexpected description format")
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Warning: Could not parse description as JSON: {e}")
        image_data = {
            "subject": {"type": "general"},
            "technical": {"resolution": {"width": 512, "height": 512}}
        }

    subject_type = image_data.get('subject', {}).get('type', 'general')
    width = image_data.get('technical', {}).get('resolution', {}).get('width', 512)
    height = image_data.get('technical', {}).get('resolution', {}).get('height', 512)
    aspect_ratio = round(width / height, 2)

    prompt = f"""Create an image with:
        Content based on this description:
{original_description}

        CRITICAL REQUIREMENTS: The characters "{random_key}" MUST be prominently and clearly visible in the image while appearing naturally integrated. These characters should be:
        - As large as possible while maintaining natural integration with the scene
        - Must positioned where they will be clearly visible and unobstructed
        - Must be the ONLY text or numbers visible in the image
        - Shown at a near-frontal angle (maximum 30-degree deviation)
        - Must not be blocked or obscured by other elements
        - Integrated naturally into the scene (e.g. as signage, displays, markings, or other contextually appropriate elements)
        - Should look like they belong in the scene, not artificially overlaid

        The integration should maintain visual coherence with the scene while ensuring "{random_key}" remains clearly visible.

        Generate the image at {width}x{height} resolution with an aspect ratio of {aspect_ratio} (width:height).

        Remember: The absolute clarity and visibility of "{random_key}" is essential - it should be easily noticeable in the image while still appearing as part of the scene. NO other text or numbers should be visible anywhere in the image."""

    return prompt

def save_image(decoded_image, output_dir, original_filename):
    """Save a single generated image"""
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"sentinel_{original_filename}")
    
    with open(output_filename, "wb") as f:
        f.write(decoded_image)
    
    return output_filename

def load_or_create_record(record_file):
    """Load or create a record file"""
    if os.path.exists(record_file):
        with open(record_file, 'r') as f:
            return json.load(f)
    return {
        "description_prompt": description_prompt,  # Save the description prompt once
        "processed_images": {}
    }

def save_record(record_file, records):
    """Save the record to a file"""
    with open(record_file, 'w') as f:
        json.dump(records, f, indent=4)

def collect_image_paths(input_dir, num_images):
    extensions = (".jpg", ".jpeg", ".png")
    if not os.path.isdir(input_dir):
        return []

    flat_images = [
        os.path.join(input_dir, fname)
        for fname in sorted(os.listdir(input_dir))
        if fname.lower().endswith(extensions)
    ]
    if flat_images:
        return flat_images[:num_images]

    image_paths = []
    for i in range(660):  # From 00000 to 00659
        folder = os.path.join(input_dir, str(i).zfill(5))
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(extensions):
                image_paths.append(os.path.join(folder, fname))
                if len(image_paths) >= num_images:
                    return image_paths

    return image_paths

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="ImageSentinel pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="../data/LLaVA_data/sentinelImages/", help="Directory to save generated images")
    parser.add_argument("--record_file", type=str, default="../data/LLaVA_data/processing_record.json", help="Path to the record file")
    parser.add_argument("--key_length", type=int, default=6, help="Length of the random key")
    parser.add_argument("--num_images", type=int, default=2, help="Number of images to process from the start of the dataset")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--base_url", type=str, default="https://api.poe.com/v1", help="OpenAI-compatible base URL")
    args = parser.parse_args()

    # Initialize the OpenAI client
    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url=args.base_url,
    )
    
    # Get the first N images in either a flat folder or the 00000/00001/... structure
    image_files = collect_image_paths(args.input_dir, args.num_images)
    
    # Load or create a record
    records = load_or_create_record(args.record_file)
    
    # Process each image
    for image_path in image_files:
        # Get the original filename
        original_filename = os.path.basename(image_path)
        
        # Check if the image has already been processed
        if original_filename in records["processed_images"]:
            print(f"Skipping already processed image: {original_filename}")
            continue
        
        try:
            # Generate a random key
            random_key = generate_random_key(length=args.key_length)
            
            # Get the image description
            description = test_image_description(client, description_prompt, [image_path])
            print(f"Processing image: {original_filename}")
            print("Generated description:", description)
            
            # Generate a prompt for the new image
            generation_prompt = generate_image_prompt(description, random_key)
            
            # Generate the new image
            decoded_images = image_generation_gpt(generation_prompt, client)
            
            if decoded_images and len(decoded_images) > 0:
                # Save the generated image
                output_path = save_image(decoded_images[0], args.output_dir, original_filename)
                
                # Update the record
                records["processed_images"][original_filename] = {
                    "random_key": random_key,
                    "original_image": image_path,
                    "generated_image": output_path,
                    "image_description": description,
                    "generation_prompt": generation_prompt,
                    "timestamp": str(datetime.datetime.now())
                }
                
                # Save the record
                save_record(args.record_file, records)
                print(f"Saved image and record: {original_filename}")
            else:
                print(f"Generation error, skipping: {original_filename}")
            
        except Exception as e:
            print(f"Error processing image {original_filename}: {str(e)}")
            continue

if __name__ == "__main__":
    # Image description prompt
    description_prompt = """Analyze this image and extract only the key visual features that define its core appearance.

    Provide your response as TEXT in valid JSON format! DO NOT generate images!
    
    Output Format Requirements:

    {
        "subject": {
            "type": <string>,  // core subject category
            "brief description": <string>  // main visual characteristics
        },
        "context": {
            "setting": <string>,  // basic environment
            "lighting": <string>,  // overall lighting condition
            "color_scheme": [<string>]  // dominant colors
        },
        "style": {
            "visual_type": <string>,  // e.g., "photograph", "painting", "digital art", "sketch", "design drawing"
            "era_characteristics": <string>,  // e.g., "modern", "vintage 80s", "contemporary", "historical"
            "photo_style": <string>,  // e.g., "professional shot", "casual snapshot", "selfie", "documentary"
            "image_quality": <string>,  // e.g., "high resolution", "grainy", "film-like", "digital sharp"
            "artistic_approach": <string>,  // e.g., "realistic", "stylized", "abstract", "minimalist"
            "overall_mood": <string>  // e.g., "candid", "formal", "artistic", "commercial"
        },
        "technical": {
            "resolution": {"width": <int>, "height": <int>},
            "image_type": <string>
        }
    }

    Note: Focus only on major visual characteristics and overall style. Capture the essence of the image while allowing creative freedom in recreation. Do not include any specific text, numbers, names or identifiable characters in the description."""
    
    main()
