"""
Trainer adapted from https://github.com/aimagelab/DiCO/blob/master/trainers/dico_trainer.py
"""

# -------------------- PATCHES
def patch_trainer_save_checkpoint():
    """
    Patch LLaVATrainer to save correctly mm_adapter weights and peft config in intermediate checkpoints.
    See:
    - https://github.com/haotian-liu/LLaVA/issues/844
    - https://github.com/haotian-liu/LLaVA/issues/729
    """
    import os
    import torch
    from llava.train.llava_trainer import get_mm_adapter_state_maybe_zero_3
    from llava.train.train import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
    import llava

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                os.makedirs(output_dir, exist_ok=True)
                state_dict = get_peft_state_maybe_zero_3(
                    self.model.named_parameters(), self.args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    self.model.named_parameters()
                )
                if self.is_world_process_zero():
                    print(f"save models to {output_dir} ")
                    self.model.config.save_pretrained(output_dir)
                    self.model.save_pretrained(output_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
            else:
                super()._save_checkpoint(model, trial, metrics)

    llava.train.llava_trainer.LLaVATrainer._save_checkpoint = _save_checkpoint
    print("### LLaVATrainer._save_checkpoint patched successfully! ###")


patch_trainer_save_checkpoint()


# -------------------- IMPORTS
import copy
import json
import os
import pathlib
import random
from argparse import Namespace
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    logging,
    set_seed,
)

from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX
from llava.mm_utils import process_anyres_image, tokenizer_image_token
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.train.llava_trainer import LLaVATrainer
from llava.train.train import (
    LazySupervisedDataset,
    ModelArguments,
    TrainingArguments,
    find_all_linear_names,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    preprocess_multimodal,
    rank0_print,
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
)
from chair_modeling import ChairRewardModel


logger = logging.get_logger(__name__)

SEED = 42
local_rank = None


# -------------------- DEBUG
class CUDATimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.elapsed_time = None

    def __enter__(self):
        self.start.record()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        self.elapsed_time = self.start.elapsed_time(self.end)


# -------------------- PARSER
@dataclass
class DataArguments:
    train_split_path: str
    eval_split_path: str
    detections_path: str
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "pad"
    mini_eval: bool = field(default=True)


@dataclass
class CustomArguments:
    dpo_beta: float = field(default=0.2)
    dico_tau: float = field(default=300)
    dpo_num_beams: int = field(default=2)
    dpo_max_new_tokens: int = field(default=200)
    online_generations: bool = field(default=False)
    dpo_xe_weight: float = field(default=0.05)
    dpo_chair_weight: float = field(default=1.0)
    dpo_recall_weight: float = field(default=0.0)


@dataclass
class LlavaSeq2SeqTrainingArguments(TrainingArguments, Seq2SeqTrainingArguments):
    pass


# -------------------- DATA PREPROCESSING - OFFLINE
def preprocess_offlinedpo(
    sources,
    completions,
    targets,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:

    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_1:
        return preprocess_offlinedpo_llama_3_1(sources=sources, completions=completions, targets=targets, tokenizer=tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_offlinedpo_v1(sources=sources, completions=completions, targets=targets, tokenizer=tokenizer, has_image=has_image)
    else:
        raise NotImplementedError("preprocess only supports models based on 'llama_3_1' and 'v1'")


def preprocess_offlinedpo_llama_3_1(
    sources,
    completions,
    targets,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])

        conv.append_message(roles["gpt"], None)
        conversations.append(conv.get_prompt())

    # Tokenize prompts

    if has_image:
        prompt_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        prompt_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # remove the first bos token
    if prompt_ids[0][0] == prompt_ids[0][1] == tokenizer.bos_token_id:
        prompt_ids = prompt_ids[:, 1:]

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_1

    # Tokenize completions
    completions = [completion + tokenizer.eos_token for completion in completions] # add EOS
    completion_ids = tokenizer(
        completions,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    completion_ids = completion_ids[:, 1:] # remove BOS

    # Tokenize targets
    targets = [target + tokenizer.eos_token for target in targets] # add EOS
    target_ids = tokenizer(
        targets,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    target_ids = target_ids[:, 1:] # remove BOS

    return dict(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        target_ids=target_ids
    )


def preprocess_offlinedpo_v1(
    sources: Sequence[str],
    completions,
    targets,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    assert conversation_lib.default_conversation.version.startswith("v1")

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []    # ["q1 a1 q2 a2 ... qn",]
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
       
        conv.append_message(roles["gpt"], None)
        conversations.append(conv.get_prompt())

    # Tokenize prompts
    if has_image:
        prompt_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        prompt_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # Tokenize completions
    completions = [completion + "</s>" for completion in completions] # add EOS
    completion_ids = tokenizer(
        completions,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    completion_ids = completion_ids[:, 1:] # remove BOS

    # Tokenize targets
    targets = [target + "</s>" for target in targets] # add EOS
    target_ids = tokenizer(
        targets,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    target_ids = target_ids[:, 1:] # remove BOS

    return dict(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        target_ids=target_ids
    )


class OfflineDPODataset(LazySupervisedDataset):
    MINI_EVAL_SIZE = 500
    
    def __init__(
        self, 
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        train_eval,
        model_args
    ):
        super(OfflineDPODataset, self).__init__(data_path, tokenizer, data_args, model_args)
        self.train_eval = train_eval
        if data_args.mini_eval and train_eval == "eval":
            self.list_data_dict = self.list_data_dict[:self.MINI_EVAL_SIZE]
            logger.warning("Using mini eval")
    

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.data_args.image_aspect_ratio == "anyres":
                image_size = image.size
                image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints, self.siglip) # torch.Size([5, 3, 336, 336])
            else:
                image_size = image.size
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        assert isinstance(i, int), "expected an int i"
        completions = [self.list_data_dict[i]["answer_1"], self.list_data_dict[i]["answer_2"]]
        targets = [self.list_data_dict[i]["target"]] # list containing a single str
        data_dict = preprocess_offlinedpo(
            sources=sources,
            completions=completions,
            targets=targets,
            tokenizer=self.tokenizer,
            has_image=('image' in self.list_data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(
                prompt_ids=data_dict["prompt_ids"][0],
                completion_ids=data_dict["completion_ids"],
                target_ids=data_dict["target_ids"][0]
            )
            
        # store sample ids to retrieve detections
        assert isinstance(i, int)
        data_dict["sample_id"] = self.list_data_dict[i]["id"]

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class OfflineDPODataCollator:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        prompt_ids, completion_ids, target_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("prompt_ids", "completion_ids", "target_ids")
        )

        prompt_ids = torch.nn.utils.rnn.pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        completion_ids = [c for pair in completion_ids for c in pair]
        completion_ids = torch.nn.utils.rnn.pad_sequence(
            completion_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        target_ids = torch.nn.utils.rnn.pad_sequence(
            target_ids,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        prompt_ids = prompt_ids[:, :self.tokenizer.model_max_length]
        completion_ids = completion_ids[:, :self.tokenizer.model_max_length]
        target_ids = target_ids[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            prompt_ids=prompt_ids,
            attention_mask=prompt_ids.ne(self.tokenizer.pad_token_id),
            completion_ids=completion_ids,
            target_ids=target_ids,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # add sample ids
        batch["sample_ids"] = [instance["sample_id"] for instance in instances]

        return batch


def make_offlinedpo_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    model_args
) -> Dict:
    train_dataset = OfflineDPODataset(
        tokenizer=tokenizer,
        data_path=data_args.train_split_path,
        data_args=data_args,
        train_eval="train",
        model_args=model_args
    )
    eval_dataset = OfflineDPODataset(
        tokenizer=tokenizer,
        data_path=data_args.eval_split_path,
        data_args=data_args,
        train_eval="eval",
        model_args=model_args
    )
    data_collator = OfflineDPODataCollator(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


# -------------------- TRAINER
def _get_batch_xe(logits, labels):
    """
    For each batch and for each completion computes XE with the corresponding target.

    Args:
        logits: (B * num_completions, prompt_completion_len, vocab_size)
        labels: (B * num_completions, prompt_completion_len)

    Returns:
        xe: (B * num_completions,)
    """
    assert logits.shape[:-1] == labels.shape
    B_num_completions, _, vocab_size = logits.shape

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    xe = F.cross_entropy(shift_logits, shift_labels, reduction="none") # (B * num_completions * (prompt_completion_len-1),)
    
    mask = (shift_labels != IGNORE_INDEX).float()               # 1s where loss is active
    xe = xe.view(B_num_completions, -1)                         # (B * num_completions, prompt_completion_len-1)
    xe = (xe * mask.view(B_num_completions, -1)).sum(dim=-1)
    count = mask.view(B_num_completions, -1).sum(dim=-1)        # (B * num_completions,)
    xe = xe / (count + 1e-8)
    return xe


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != IGNORE_INDEX)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == IGNORE_INDEX] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)   


class CHAIRDPOTrainer(LLaVATrainer, Seq2SeqTrainer):
    def __init__(self, *, reward_model: ChairRewardModel, tokenizer_rm = None, custom_args: Optional[Namespace] = None, **kwargs):
        super().__init__(**kwargs)
        self.custom_args = custom_args

        assert reward_model is not None

        self.reward_model = reward_model
        self.tokenizer_rm = tokenizer_rm # TODO: remove tokenizer_rm arg

        # --- generation args
        self.eval_gen_kwargs = dict(
            max_new_tokens=self.custom_args.dpo_max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )

        if self.args.per_device_eval_batch_size > 1: 
            logger.warning("WARNING: using per_device_eval_batch_size > 1 needs batched generation with right pad, this could produce inaccurate completions")

        # --- metrics
        self._stored_metrics = defaultdict(lambda: defaultdict(list))


    def compute_rank(self, rewards, num_beams):
        rewards_batch = rewards.reshape(-1, num_beams)
        positive_indexes = torch.argmin(rewards_batch, dim=-1, keepdim=True) # low chair --> few hallucinations
        positive_mask = torch.zeros_like(rewards_batch, dtype=torch.bool)

        return torch.scatter(
            positive_mask,
            dim=1,
            index=positive_indexes,
            src=torch.ones_like(positive_indexes, dtype=torch.bool)
        )
    

    def weight_quality_distances(self, rewards, policy_rejected_logps, reference_rejected_logps, chosen_mask, rejected_mask, tau):
        max_rewards = rewards.view_as(chosen_mask)[chosen_mask].unsqueeze(1)
        num_negatives = chosen_mask.shape[1]
        diff_measure = (max_rewards - rewards.reshape(-1, num_negatives))[rejected_mask] * tau
        gamma = torch.nn.functional.softmax(diff_measure.reshape(-1, num_negatives-1).cuda(), dim=-1).reshape(-1)
        policy_rejected_logps *= gamma
        reference_rejected_logps *= gamma

        return policy_rejected_logps, reference_rejected_logps


    def dpo_loss(self, candidate_policy, ref_policy, chosen_mask, rejected_mask, rewards, xe, metrics_prefix):
        bsz = chosen_mask.shape[0]

        reference_chosen_logps = ref_policy[chosen_mask]
        reference_rejected_logps = ref_policy[rejected_mask]

        policy_chosen_logps = candidate_policy[chosen_mask]
        policy_rejected_logps = candidate_policy[rejected_mask]

        policy_rejected_logps, reference_rejected_logps = self.weight_quality_distances(rewards, policy_rejected_logps, reference_rejected_logps, chosen_mask, rejected_mask, self.custom_args.dico_tau)

        policy_rejected_logps = policy_rejected_logps.view(bsz, -1).sum(-1)
        reference_rejected_logps = reference_rejected_logps.view(bsz, -1).sum(-1)
            
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        dpo_term = -F.logsigmoid(self.custom_args.dpo_beta * logits)
        xe_term = self.custom_args.dpo_xe_weight * xe
        loss = dpo_term + xe_term
        chosen_rewards = self.custom_args.dpo_beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.custom_args.dpo_beta * (policy_rejected_logps - reference_rejected_logps).detach()

        loss = loss.mean()

        # --- metrics
        metrics = {}
        gathered_chosen_rewards = self.accelerator.gather(chosen_rewards)
        metrics[f"{metrics_prefix}_rewards/chosen"] = gathered_chosen_rewards.mean().item()
        gathered_rejected_rewards = self.accelerator.gather(rejected_rewards)
        metrics[f"{metrics_prefix}_rewards/rejected"] = gathered_rejected_rewards.mean().item()
        margin = gathered_chosen_rewards - gathered_rejected_rewards
        metrics[f"{metrics_prefix}_rewards/margins"] = margin.mean().item()
        accuracy = margin > 0
        metrics[f"{metrics_prefix}_rewards/accuracies"] = accuracy.float().mean().item()
        
        # halscore is weight_chair * chair - weight_recall * recall, lower is better
        gathered_halscore_chosen = self.accelerator.gather(rewards.view_as(chosen_mask)[chosen_mask])
        gathered_halscore_rejected = self.accelerator.gather(rewards.view_as(chosen_mask)[rejected_mask])
        metrics[f"{metrics_prefix}_halscore/chosen"] = gathered_halscore_chosen.mean().item()
        metrics[f"{metrics_prefix}_halscore/rejected"] = gathered_halscore_rejected.mean().item()

        gathered_xe = self.accelerator.gather(xe.detach())
        metrics[f"{metrics_prefix}/dataset_completion_mean_xe"] = gathered_xe.mean().item()

        # loss components
        gathered_dpo_term = self.accelerator.gather(dpo_term.detach())
        metrics[f"{metrics_prefix}/dpo_term"] = gathered_dpo_term.mean().item()

        gathered_xe_term = self.accelerator.gather(xe_term.detach())
        metrics[f"{metrics_prefix}/xe_term"] = gathered_xe_term.mean().item()

        return loss, metrics


    def get_batch_loss_metrics_offline(self, model, inputs, train_eval):
        #                       | train                 | eval                  |
        # model                 | DeepSpeedEngine       | PeftModelForCausalLM  |        
        # self.model            | PeftModelForCausalLM  | PeftModelForCausalLM  |
        # self.model_wrapped    | DeepSpeedEngine       | DeepSpeedEngine       |
        # self.deepspeed        | DeepSpeedEngine       | DeepSpeedEngine       |

        prompt_ids = inputs.pop("prompt_ids")               # (B, prompt_len)
        prompt_mask = inputs.pop("attention_mask")          # (B, prompt_len)
        images = inputs.pop("images")                       # (B, C, H, W)
        completion_ids = inputs.pop("completion_ids")       # (B * num_completions, completion_len)
        num_completions = self.custom_args.dpo_num_beams        # completions per prompt

        # 1. Rank completions with reward model
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        sample_ids = inputs.pop("sample_ids")
        sample_ids = [sample_id for sample_id in sample_ids for _ in range(num_completions)]

        rewards = self.reward_model(
            caption_texts=completion_texts,
            sample_ids=sample_ids
        ) 

        chair = torch.tensor([r["chair_i"] for r in rewards], device=completion_ids.device) # (B * num_completions,)
        recall = torch.tensor([r["recall"] for r in rewards], device=completion_ids.device) # (B * num_completions,)

        rewards = self.custom_args.dpo_chair_weight * chair - self.custom_args.dpo_recall_weight * recall # (B * num_completions,)

        chosen_mask = self.compute_rank(rewards=rewards, num_beams=num_completions) # (B, num_completions)
        rejected_mask = ~chosen_mask

        # 2. Build inputs by concatenating prompts and completions
        interleaved_prompt_lengths = prompt_mask.sum(dim=-1).repeat_interleave(num_completions, dim=0)
        completion_lengths = (completion_ids != self.tokenizer.pad_token_id).sum(dim=-1)
        max_length = torch.max(interleaved_prompt_lengths + completion_lengths).item()

        prompt_completion_ids = torch.full(
            (prompt_ids.size(0) * num_completions, max_length), 
            fill_value=self.tokenizer.pad_token_id, 
            device=prompt_ids.device, 
            dtype=prompt_ids.dtype
        )
        ref_labels = torch.full_like(prompt_completion_ids, fill_value=IGNORE_INDEX) # contains completion_ids, will be used as label to compute logprobs

        for i, (plen, clen) in enumerate(zip(interleaved_prompt_lengths, completion_lengths)):
            prompt_completion_ids[i, :plen] = prompt_ids[i // num_completions, :plen]
            prompt_completion_ids[i, plen:plen+clen] = completion_ids[i, :clen]
            ref_labels[i, plen:plen+clen] = completion_ids[i, :clen]

        if prompt_completion_ids.size(-1) > self.tokenizer.model_max_length:
            logger.warning("prompt_completion_ids exceeds model_max_length by {}".format(prompt_completion_ids.size(-1)-self.tokenizer.model_max_length))
            prompt_completion_ids = prompt_completion_ids[:, :self.tokenizer.model_max_length]
            ref_labels = ref_labels[:, :self.tokenizer.model_max_length]

        prompt_completion_mask = prompt_completion_ids != self.tokenizer.pad_token_id

        interleaved_images = images.repeat_interleave(num_completions, dim=0)
        del images, prompt_ids, prompt_mask, completion_ids

        # 3. Compute logprob of the completion for the ref model given the prompt and the image
        with torch.no_grad():
            with self.model.disable_adapter():
                (
                    _, 
                    _, 
                    embeds_attention_mask, 
                    _, 
                    inputs_embeds, 
                    ref_labels # refined to take into account image token embeds
                ) = self.model.prepare_inputs_labels_for_multimodal(
                    input_ids=prompt_completion_ids, 
                    position_ids=None, 
                    attention_mask=prompt_completion_mask, 
                    past_key_values=None, 
                    labels=ref_labels,
                    images=interleaved_images, 
                    image_sizes=None
                )

                ref_logits = self.model(
                    input_ids=None,
                    attention_mask=embeds_attention_mask,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=inputs_embeds,
                    labels=None,
                    use_cache=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    images=None,
                    image_sizes=None,
                    return_dict=None,
                ).logits # (B * num_completions, prompt_completion_len, vocab_size)

            del embeds_attention_mask, inputs_embeds

            ref_policy = _get_batch_logps(
                logits=ref_logits, 
                labels=ref_labels, 
                average_log_prob=False
            ).view(chosen_mask.shape) # (B, num_completions)

        # 4. Compute logprob of the completion for the candidate model given the prompt and the image
        target_ids = inputs.pop("target_ids") # (B, target_len)
        target_ids = target_ids.repeat_interleave(num_completions, dim=0) # (B * num_completions, target_len)
        target_labels = torch.full_like(prompt_completion_ids, fill_value=IGNORE_INDEX) # contains target_ids, will be used as label to compute XE
        target_lengths = (target_ids != IGNORE_INDEX).sum(dim=-1)

        # NOTE to use prepare_inputs_labels_for_multimodal we need target_labels.shape == prompt_completion_ids.shape
        # if the unpadded length of prompt_i + unpadded length of target_i < prompt_completion_ids.size(-1) we right pad target_i with IGNORE_INDEX
        # if the unpadded length of prompt_i + unpadded length of target_i > prompt_completion_ids.size(-1) we truncate right target_i
        for i, (plen, tlen) in enumerate(zip(interleaved_prompt_lengths, target_lengths, strict=True)):
            if plen+tlen > prompt_completion_ids.size(-1):
                tlen = prompt_completion_ids.size(-1) - plen

            target_labels[i, plen:plen+tlen] = target_ids[i, :tlen]

        (
            _, 
            _, 
            embeds_attention_mask, 
            _, 
            inputs_embeds, 
            target_labels # refined to take into account image token embeds
        ) = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=prompt_completion_ids, 
            position_ids=None, 
            attention_mask=prompt_completion_mask, 
            past_key_values=None, 
            labels=target_labels,
            images=interleaved_images, 
            image_sizes=None
        )

        candidate_logits = self.model(
            input_ids=None,
            attention_mask=embeds_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            images=None,
            image_sizes=None,
            return_dict=None,
        ).logits # (B * num_completions, prompt_completion_len, vocab_size)

        # compute for each batch the mean XE of completions wrt targets
        xe = _get_batch_xe(candidate_logits, target_labels) # (B * num_completions,)
        xe = xe.view(chosen_mask.shape) # (B, num_completions)
        xe = xe.mean(axis=-1) # (B,)

        del embeds_attention_mask, inputs_embeds, prompt_completion_ids

        candidate_policy = _get_batch_logps(
            logits=candidate_logits, 
            labels=ref_labels, 
            average_log_prob=False
        ).view(chosen_mask.shape) # (B, num_completions)

        # 5. Compute DPO loss
        loss, metrics = self.dpo_loss(
            candidate_policy=candidate_policy,
            ref_policy=ref_policy, 
            chosen_mask=chosen_mask, 
            rejected_mask=rejected_mask, 
            rewards=rewards,
            xe=xe,
            metrics_prefix=train_eval
        )

        # Update metrics with chair and recall
        gathered_chair_chosen = self.accelerator.gather(chair.view_as(chosen_mask)[chosen_mask])
        gathered_chair_rejected = self.accelerator.gather(chair.view_as(chosen_mask)[rejected_mask])
        gathered_chair = self.accelerator.gather(chair)
        metrics[f"{train_eval}_chair/chosen"] = gathered_chair_chosen.mean().item()
        metrics[f"{train_eval}_chair/rejected"] = gathered_chair_rejected.mean().item()
        metrics[f"{train_eval}_chair/mean"] = gathered_chair.mean().item()

        gathered_recall_chosen = self.accelerator.gather(recall.view_as(chosen_mask)[chosen_mask])
        gathered_recall_rejected = self.accelerator.gather(recall.view_as(chosen_mask)[rejected_mask])
        gathered_recall = self.accelerator.gather(recall)
        metrics[f"{train_eval}_recall/chosen"] = gathered_recall_chosen.mean().item()
        metrics[f"{train_eval}_recall/rejected"] = gathered_recall_rejected.mean().item()
        metrics[f"{train_eval}_recall/mean"] = gathered_recall.mean().item()

        return loss, metrics


    def store_metrics(self, metrics, train_eval):
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)


    def compute_loss(self, model, inputs, return_outputs=False):
        do_profiling = False # TODO: parser arg

        ctx = CUDATimer if do_profiling else nullcontext

        with ctx() as timer:
            loss, metrics = self.get_batch_loss_metrics_offline(model, inputs, train_eval="train")

        if do_profiling and self.is_world_process_zero():
            print("forward elapsed ms:", timer.elapsed_time)
            metrics["forward_elapsed_ms"] = timer.elapsed_time

        if self.is_world_process_zero():
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics

        return loss


    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # reset counters for overall chair-i and overall recall
        self.hallucinated_word_count    = torch.tensor(0.0, device=self.accelerator.device)
        self.coco_word_count            = torch.tensor(0.0, device=self.accelerator.device)
        self.num_recall_gt_objects      = torch.tensor(0.0, device=self.accelerator.device)
        self.num_gt_objects             = torch.tensor(0.0, device=self.accelerator.device)

        # reset containers for per sample chair-i and per sample recall
        self.sample_chairi_list = []
        self.sample_recall_list = []

        # reset container for XE
        self.xe_list = []

        # reset container for completions
        self.prompt_completion_list = []

        # start eval loop
        output = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        # gather and log all metrics
        logs = {}
        
        gathered_hallucinated_word_count = self.accelerator.gather(self.hallucinated_word_count)
        gathered_coco_word_count = self.accelerator.gather(self.coco_word_count)

        total_hallucinated_word_count = gathered_hallucinated_word_count.sum().item()
        total_coco_word_count = gathered_coco_word_count.sum().item()

        chair_i_overall = 0.0
        if total_coco_word_count > 0.0:
            chair_i_overall = total_hallucinated_word_count / total_coco_word_count
        
        logs[f"{metric_key_prefix}_chair_i_overall"] = chair_i_overall

        gathered_num_recall_gt_objects = self.accelerator.gather(self.num_recall_gt_objects)
        gathered_num_gt_objects = self.accelerator.gather(self.num_gt_objects)

        total_num_recall_gt_objects = gathered_num_recall_gt_objects.sum().item()
        total_num_gt_objects = gathered_num_gt_objects.sum().item()

        recall_overall = 0.0
        if total_num_gt_objects > 0.0:
            recall_overall = total_num_recall_gt_objects / total_num_gt_objects
        
        logs[f"{metric_key_prefix}_recall_overall"] = recall_overall

        # gather per sample containers
        gathered_sample_chairi_list = self.accelerator.gather(torch.tensor(self.sample_chairi_list, device=self.accelerator.device)) 
        gathered_sample_recall_list = self.accelerator.gather(torch.tensor(self.sample_recall_list, device=self.accelerator.device))
        
        mean_sample_chairi = gathered_sample_chairi_list.mean().item()
        mean_sample_recall = gathered_sample_recall_list.mean().item()

        logs[f"{metric_key_prefix}_mean_sample_chairi"] = mean_sample_chairi
        logs[f"{metric_key_prefix}_mean_sample_recall"] = mean_sample_recall

        # gather XE
        gathered_xe_list = self.accelerator.gather(torch.tensor(self.xe_list, device=self.accelerator.device))
        mean_xe = gathered_xe_list.mean().item()
        logs[f"{metric_key_prefix}_generated_completion_mean_xe"] = mean_xe

        # save prompt_completion_list to file
        # we do not gather on purpose
        if self.is_world_process_zero():
            print("Saving prompt completion data...", end=" ")
            with open(os.path.join(self.args.output_dir, "prompt_completion.jsonl"), mode="a") as f:
                f.write(json.dumps(self.prompt_completion_list) + "\n")
            print("Done.")

        # only the main process logs to wandb
        if self.is_world_process_zero():
            super().log(logs)

        return output


    def update_chair_xe(self, model, inputs):
        # 1. Compute chair related quantities
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["attention_mask"]
        images = inputs["images"]
        sample_ids = inputs["sample_ids"]

        with torch.no_grad():
            output = model.generate(
                inputs=prompt_ids, 
                attention_mask=prompt_mask,
                images=images,
                **self.eval_gen_kwargs
            )
            #completion_ids = output.sequences[:, 1:] # skip BOS # not needed in transformers 4.43
            completion_ids = output.sequences

        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        if len(self.prompt_completion_list) < 10:
            placeholder_tok_id = self.tokenizer("#").input_ids[-1]
            temp = torch.where(prompt_ids < 0, torch.ones_like(prompt_ids)*placeholder_tok_id, prompt_ids) # HACK to decode image token as #
            temp = self.tokenizer.batch_decode(temp, skip_special_tokens=True)
            self.prompt_completion_list.append(
                dict(step=self.state.global_step, 
                     prompt=temp,
                     completion=completion_texts)
            )

        chair_dicts = [
            self.reward_model.chair.compute_chairi_sample(completion_text, sample_id) 
            for completion_text, sample_id in zip(completion_texts, sample_ids, strict=True)
        ]

        for chair_dict in chair_dicts:
            self.hallucinated_word_count    += torch.tensor(float(chair_dict["partial_hallucinated_word_count"]), device=self.accelerator.device)
            self.coco_word_count            += torch.tensor(float(chair_dict["partial_coco_word_count"]), device=self.accelerator.device)
            self.num_recall_gt_objects      += torch.tensor(float(chair_dict["partial_num_recall_gt_objects"]), device=self.accelerator.device)
            self.num_gt_objects             += torch.tensor(float(chair_dict["partial_num_gt_objects"]), device=self.accelerator.device)

            self.sample_chairi_list.append(chair_dict["chair_i"])
            self.sample_recall_list.append(chair_dict["recall"])

        
        # 2. Compute cross entropy between generated completions and targets
        prompt_lenghts = prompt_mask.sum(dim=-1)
        completion_lengths = (completion_ids != self.tokenizer.pad_token_id).sum(dim=-1)
        max_length = torch.max(prompt_lenghts + completion_lengths).item()

        prompt_completion_ids = torch.full(
            (prompt_ids.size(0), max_length), 
            fill_value=self.tokenizer.pad_token_id, 
            device=prompt_ids.device, 
            dtype=prompt_ids.dtype
        )

        for i, (plen, clen) in enumerate(zip(prompt_lenghts, completion_lengths, strict=True)):
            prompt_completion_ids[i, :plen] = prompt_ids[i, :plen]
            prompt_completion_ids[i, plen:plen+clen] = completion_ids[i, :clen]

        if prompt_completion_ids.size(-1) > self.tokenizer.model_max_length:
            logger.warning("prompt_completion_ids exceeds model_max_length by {}".format(prompt_completion_ids.size(-1)-self.tokenizer.model_max_length))
            prompt_completion_ids = prompt_completion_ids[:, :self.tokenizer.model_max_length]

        prompt_completion_mask = prompt_completion_ids != self.tokenizer.pad_token_id
        
        target_ids = inputs["target_ids"]
        target_labels = torch.full_like(prompt_completion_ids, fill_value=IGNORE_INDEX) # contains target_ids, will be used as label to compute XE
        target_lengths = (target_ids != IGNORE_INDEX).sum(dim=-1)

        # if the unpadded length of prompt_i + unpadded length of target_i < prompt_completion_ids.size(-1) we right pad target_i with IGNORE_INDEX
        # if the unpadded length of prompt_i + unpadded length of target_i > prompt_completion_ids.size(-1) we truncate right target_i
        for i, (plen, tlen) in enumerate(zip(prompt_lenghts, target_lengths, strict=True)):
            if plen+tlen > prompt_completion_ids.size(-1):
                tlen = prompt_completion_ids.size(-1) - plen

            target_labels[i, plen:plen+tlen] = target_ids[i, :tlen]

        del target_ids

        with torch.no_grad():
            xe = model(
                input_ids=prompt_completion_ids,
                attention_mask=prompt_completion_mask,
                labels=target_labels, # allow to compute XE
                images=images
            ).loss.item()
        self.xe_list.append(xe)


    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys,
    ):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                self.update_chair_xe(model, inputs) # NOTE: call this function before since the other alters inputs
                loss, metrics = self.get_batch_loss_metrics_offline(model, inputs, train_eval="eval")
        
        if self.is_world_process_zero():
            self.store_metrics(metrics, train_eval="eval")

        assert prediction_loss_only
        if prediction_loss_only:
            return (loss.detach(), None, None)

    
    def log(self, logs: Dict[str, float]) -> None:
        if not self.is_world_process_zero():
            return

        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"

        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        return super().log(logs)


# -------------------- MAIN
def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, LlavaSeq2SeqTrainingArguments, CustomArguments)
    )
    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    assert model_args.vision_tower is not None
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # --- handle special tokens
    # for all the version of llama 3 not expand the dictionary with unk token
    # it can create problem in stage two when importing the configuration value of the vocab size
    if "llama_3" not in training_args.llm_backbone:
        if tokenizer.unk_token is None:
            print("resize embedding dimension")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(unk_token="[UNK]"),
                tokenizer=tokenizer,
                model=model,
            )
    # select the correct PAD token for llama_3_1 and llama_3
    if training_args.llm_backbone == "llama_3_1":
        print(f"pad token: {training_args.llm_pad_token}")
        if training_args.llm_pad_token == 'end_of_text':
            tokenizer.pad_token_id= 128001
        elif training_args.llm_pad_token == 'eot':
            tokenizer.pad_token_id= 128009
        elif training_args.llm_pad_token == 'pad':
            tokenizer.pad_token_id= 128004
        else:
            raise ValueError(f"Unknown llm_pad_token")
        
    elif training_args.llm_backbone == "llama_3":
        if training_args.llm_pad_token == 'eos':
            tokenizer.pad_token = tokenizer.eos_token
        elif training_args.llm_pad_token == 'pad':
            tokenizer.pad_token_id= 128003
        else:
            tokenizer.pad_token = tokenizer.unk_token

    else:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    # ---

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_aspect_ratio == 'anyres':
            base_size = vision_tower.config.image_size
            grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3]]
            model.config.image_grid_pinpoints = data_args.image_grid_pinpoints = [
                [g[0]*base_size, g[1]*base_size] for g in grids]
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    assert training_args.lora_enable
    if local_rank in [-1, 0]:
        model.print_trainable_parameters()

    reward_model = ChairRewardModel(data_args.detections_path)
    tokenizer_rm = None

    if custom_args.online_generations:
        raise NotImplementedError("Online generation is not implemented.")
    else:
        data_module = make_offlinedpo_data_module(
            tokenizer=tokenizer,
            data_args=data_args,
            model_args=model_args
        )
    
    trainer = CHAIRDPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        tokenizer_rm=tokenizer_rm,
        args=training_args,
        custom_args=custom_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
        

if __name__ == "__main__":
    random.seed(SEED)
    set_seed(SEED)
    train(attn_implementation="flash_attention_2")