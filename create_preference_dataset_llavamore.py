"""
Adapted from model_vqa.py
"""

import argparse
import torch
import os
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import orjson
import random
from llava import conversation as conversation_lib


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def supports_flash_attention():
    if not torch.cuda.is_available():
        return False

    major, _ = torch.cuda.get_device_capability()

    # Flash attention typically requires compute capability 8.0+
    return major >= 8


def main(args):
    SEED = 42
    random.seed(SEED)

    use_flash_attn = False
    if args.use_flash_attn:
        if supports_flash_attention():
            use_flash_attn = True
        else:
            print("WARNING: Your GPU does not support Flash Attention, generation won't use it.")

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path, 
        model_base=args.model_base, 
        model_name=model_name, 
        use_flash_attn=use_flash_attn
    )
    
    device = next(model.parameters()).device

    # --- load llavainstruct mix dataset
    with open(args.dataset_split_path, "rb") as f:
        dataset_split = orjson.loads(f.read())

    data_chunk = get_chunk(lst=dataset_split, n=args.num_chunks, k=args.chunk_idx)

    print("CHUNK_IDX:", args.chunk_idx)
    print("NUM_CHUNKS:", args.num_chunks)
    print("CHUNK_SIZE:", len(data_chunk), math.ceil(len(dataset_split) / args.num_chunks))

    output_chunk = []
    conv = conversation_lib.conv_llava_llama_3_1.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    for i, entry in tqdm(enumerate(data_chunk), desc="Generating preferences..."):
        selected_num_rounds = 1 if "eval" in args.dataset_split_path else random.randint(1, len(entry["conversations"]) // 2)
        num_sentences = selected_num_rounds * 2 # each round has 2 sentences
        selected_sentences = entry["conversations"][:num_sentences]

        if selected_sentences[-1]["from"] != "gpt":
            print("WARNING: last sentence is not a gpt answer")
        target = selected_sentences[-1]["value"]
        selected_sentences = selected_sentences[:-1] # we don't want the answer of the selected round to be included in the prompt

        conv.messages = []
        for j, sentence in enumerate(selected_sentences):
            role = roles[sentence["from"]]
            if role != conv.roles[j % 2]:
                print(f"wrong rounds in entry {i}")
            conv.append_message(role, sentence["value"])
        conv.append_message(roles["gpt"], None)
        prompt = conv.get_prompt()

        prompt_ids = tokenizer_image_token(
            prompt=prompt, 
            tokenizer=tokenizer, 
            image_token_index=IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).to(device)

        image_file = entry["image"]
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            completion_ids = model.generate(
                prompt_ids,
                attention_mask=(prompt_ids != tokenizer.pad_token_id),
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                num_return_sequences=args.num_completions,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        completions = [c.strip() for c in completions]

        # --- update output chunk
        output_entry = dict(
            id=entry["id"],
            image=entry["image"],
            conversations=selected_sentences,
            target=target,
            answer_1=completions[0],
            answer_2=completions[1]
        )
        output_chunk.append(output_entry)

    # --- save the chunk of the preference dataset
    if len(output_chunk) != len(data_chunk):
        print(f"WARNING: length of output chunk {len(output_chunk)} != lengh of data chunk {len(data_chunk)}")

    with open(args.output_path, "wb") as f:
        f.write(orjson.dumps(output_chunk))
    
    print("Saved preference dataset to {}".format(args.output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--dataset_split_path", type=str)
    parser.add_argument("--conv-mode", type=str, default="llama_3_1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_completions", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--use-flash-attn", action='store_true')
    args = parser.parse_args()

    main(args)
