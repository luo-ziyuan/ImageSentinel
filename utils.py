import base64

def convert_res_to_captions(res):
    captions = [c.strip() for c in res.split("\n") if c != ""]
    for i in range(len(captions)):
        if captions[i][0].isnumeric() and captions[i][1] == ".":
            captions[i] = captions[i][2:]
        elif captions[i][0] == "-":
            captions[i] = captions[i][1:]
        elif f"{i+1}." in captions[i]:
            captions[i] = captions[i][captions[i].find(f"{i+1}.")+len(f"{i+1}."):]

        captions[i] = captions[i].strip().replace("'", "").replace('"', '')
    return captions

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def message_gpt(msg, client, image_paths=[], context_msgs=[], images_idx=-1, temperature=0):
    messages = [{"role": "user",
                 "content": [{"type": "text", "text": msg}]
                 }]
    if context_msgs:
        messages = context_msgs + messages

    if image_paths:
        base_64_images = [encode_image(image_path) for image_path in image_paths]
        for i, img in enumerate(base_64_images):
            messages[images_idx]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{image_paths[i][image_paths[i].rfind('.') + 1:]};base64,{img}"}})

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "text"},
        temperature=temperature # for less randomness
    )

    res_text = res.choices[0].message.content
    return res_text

def message_gpt_w_error_handle(msg, client, image_paths, context_msgs, max_tries=3):
    unable = True
    temp = 0
    while unable and max_tries > 0:
        concepts = message_gpt(msg, client, image_paths, context_msgs=context_msgs, images_idx=0)
        print("concepts from images", concepts)

        if "unable" not in concepts and "can't" not in concepts:  # TODO make more generic
            unable = False

        temp += 1 / max_tries
        max_tries -= 1

    if unable:
        print("was unable to generate concepts, using prompt as caption")
        return ""

    return concepts

def retrieval_caption_generation_bool(prompt, image_paths, gpt_client, k_captions_per_concept=1, k_concepts=-1, decision=True, only_rephrase=False):
    if len(image_paths) > 1:
        msg1 = f'Does the second image match the instruction "{prompt}" applied over the first one? consider both content and style aspects. only answer yes or no.'
    else:
        msg1 = f'Does this image match the prompt "{prompt}"? consider both content and style aspects. only answer yes or no.'

    ans = message_gpt(msg1, gpt_client, image_paths)
    if 'yes' in ans.lower():
        return True
    else:
        return False

def retrieval_caption_generation(prompt, image_paths, gpt_client, k_captions_per_concept=1, k_concepts=-1, decision=True, only_rephrase=False):
    if decision:
        if len(image_paths) > 1:
            msg1 = f'Does the second image match the instruction "{prompt}" applied over the first one? consider both content and style aspects. only answer yes or no.'
        else:
            msg1 = f'Does this image match the prompt "{prompt}"? consider both content and style aspects. only answer yes or no.'

        ans = message_gpt(msg1, gpt_client, image_paths)
        if 'yes' in ans.lower():
            return True

        context_msgs = [{"role": "user",
                         "content": [{"type": "text", "text": msg1}]
                         },
                        {"role": "assistant",
                         "content": [{"type": "text", "text": ans}]
                         }]

        print(f"Answer was {ans}. Running imageRAG")
        if only_rephrase:
            rephrased_prompt = get_rephrased_prompt(prompt, gpt_client, image_paths, context_msgs=context_msgs, images_idx=0)
            print("rephrased_prompt:", rephrased_prompt)
            return rephrased_prompt

        msg2 = 'What are the differences between this image and the required prompt? in your answer only provide missing concepts in terms of content and style, each in a separate line. For example, if the prompt is "An oil painting of a sheep and a car" and the image is a painting of a car but not an oil painting, the missing concepts will be:\noil painting style\na sheep'
        if k_concepts > 0:
            msg2 += f'Return up to {k_concepts} concepts.'

        concepts = message_gpt_w_error_handle(msg2, gpt_client, image_paths, context_msgs, max_tries=3)
        if concepts == "":
            return prompt
    else:  # generation mode
        context_msgs = []
        msg2 = (
            f'What visual concepts does a generative model need to know to generate an image described by the prompt "{prompt}"?\n'
            'The concepts should be things like objects that should appear in the image, the style of it, etc.'
            'For example, if the prompt is "An elephant standing on a ball", 2 concepts would be: elephant, ball.'
            'In your answer only provide the concepts, each in a separate line.')

        concepts = message_gpt_w_error_handle(msg2, gpt_client, image_paths, context_msgs, max_tries=3)
        if concepts == "":
            return prompt

    print(f'retrieved concepts: {concepts}')

    msg3 = (f'For each concept you suggested above, please suggest {k_captions_per_concept} image captions describing images that explain this concept only. '
            f'The captions should be stand-alone description of the images, assuming no knowledge of the given images and prompt, that I can use to lookup images with automatically. '
            f'In your answer only provide the image captions, each in a new line with nothing else other than the caption.')
    context_msgs += [{"role": "user",
                      "content": [{"type": "text", "text": msg2}]
                      },
                     {"role": "assistant",
                      "content": [{"type": "text", "text": concepts}]
                      }]
    captions = message_gpt(msg3, gpt_client, image_paths, context_msgs=context_msgs, images_idx=0)
    return captions

def get_rephrased_prompt(prompt, gpt_client, image_paths=[], context_msgs=[], images_idx=-1):
    if not context_msgs:
        msg = f'Please rephrase the following prompt to make it clearer for a text-to-image generation model. If it\'s already clear, return it as it is. In your answer only provide the prompt and nothing else, and don\'t change the original meaning of the prompt. If it contains rare words, change the words to a description of their meaning. The prompt to be rephrased: "{prompt}"'
    else:
        msg = f'Please rephrase the following prompt to make it easier and clearer for the text-to-image generation model that generated the above image for this prompt. The goal is to generate an image that matches the given text prompt. If the prompt is already clear, return it as it is. Simplify and shorten long descriptions of known objects/entities but DO NOT change the original meaning of the text prompt. If the prompt contains rare words, change those words to a description of their meaning. In your answer only provide the prompt and nothing else. The prompt to be rephrased: "{prompt}"'

    ans = message_gpt(msg, gpt_client, image_paths, context_msgs=context_msgs, images_idx=images_idx)
    return ans.strip().replace('"', '').replace("'", '')