#!/bin/bash
#SBATCH --job-name=llava_v1.5_7b_chairdpo_lora
#SBATCH --output=path/to/logs/outputs/%x-%j
#SBATCH --error=path/to/logs/errors/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=32
#SBATCH --partition=boost_usr_prod
#SBATCH --account=SLURM_ACCOUNT_PLACEHOLDER
#SBATCH --nodes=2
#SBATCH --time=24:00:00

# IMPORTANT: replace all dummy paths ("path/to/...") and placeholders ("..._PLACEHOLDER")

echo "### NODELIST ###"
echo "$SLURM_JOB_NODELIST"
echo "################"

# Load required modules, e.g. anaconda, cuda, etc.
# module load cuda/11.8

source activate CONDA_ENV_NAME_PLACEHOLDER

cd path/to/CHAIR-DPO_release
export PYTHONPATH=.

export HF_HUB_CACHE=path/to/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE="offline"
export WANDB_DIR=path/to/wandb
export WANDB_ENTITY="ENTITY_PLACEHOLDER"
export WANDB_PROJECT="PROJECT_PLACEHOLDER"
export WANDB_WATCH="all"
export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export RANK=0

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export OMP_NUM_THREADS=1

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "MASTER ADDR: ${MASTER_ADDR}"
echo "MASTER PORT: ${MASTER_PORT}"

grid_id=0
lr=2e-6
dpo_xe_weight=0.0
dpo_beta=0.2
dpo_chair_weight=1.0
dpo_recall_weight=0.0

model_name="aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning"
llm_backbone="llama_3_1"
version="llama_3_1"
eval_save_steps=100
per_device_train_batch_size=2
model_max_length=4096
export TOKENIZER_PATH=${model_name}

echo "GRID ID:              ${grid_id}"
echo "LR:                   ${lr}"
echo "DPO XE WEIGHT:        ${dpo_xe_weight}"
echo "DPO BETA:             ${dpo_beta}"
echo "DPO CHAIR WEIGHT:     ${dpo_chair_weight}"
echo "DPO RECALL WEIGHT:    ${dpo_recall_weight}"

export WANDB_NOTES="${SLURM_JOB_NAME} | grid id ${grid_id} | lr ${lr} | xe_weight ${dpo_xe_weight} | beta ${dpo_beta} | chair_weight ${dpo_chair_weight} | recall_weight ${dpo_recall_weight}"

srun --exclusive -c $SLURM_CPUS_PER_TASK --mem $SLURM_MEM_PER_NODE \
torchrun \
--nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-endpoint=$MASTER_ADDR --rdzv-id=$SLURM_JOB_NAME --rdzv-backend=c10d \
train.py \
    --seed 42 \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --deepspeed scripts/dpo/zero2_dpo.json \
    --model_name_or_path ${model_name} \
    --llm_backbone ${llm_backbone} \
    --llm_pad_token "pad" \
    --version ${version} \
    --train_split_path "path/to/data/preference_more/refined_train_split_preference.json" \
    --eval_split_path "path/to/data/preference_more/eval_split_preference.json" \
    --detections_path "path/to/data/detections.json" \
    --image_folder "path/to/llavainstruct/images" \
    --vision_tower "openai/clip-vit-large-patch14-336" \
    --mm_projector_type "mlp2x_gelu" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio "pad" \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "path/to/checkpoints/${SLURM_JOB_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --eval_steps ${eval_save_steps} \
    --save_steps ${eval_save_steps} \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --run_name "${SLURM_JOB_NAME}" \
    --remove_unused_columns False \
    --dpo_max_new_tokens 200 \
    --dpo_num_beams 2 \
    --online_generations False \
    --report_to "wandb" \
    --prediction_loss_only \
    --mini_eval True \
    --dpo_xe_weight ${dpo_xe_weight} \
    --dpo_beta ${dpo_beta} \
    --dpo_chair_weight ${dpo_chair_weight} \
    --dpo_recall_weight ${dpo_recall_weight}
