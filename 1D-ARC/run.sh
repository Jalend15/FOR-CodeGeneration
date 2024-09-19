step=$2
mkdir -p ./logs/step_${step} && \

CUDA_VISIBLE_DEVICES=5 taskset -c 0-2 python main.py \
    --step 2 \
    --n_samples 10 \
    --epochs 1 \
    --ll-weight 1.5 \
    --pretrained_model "meta-llama/Meta-Llama-3-8B" \
    --reward_temp_end 2 \
    --pf_temp_start 4 \
    --p_buffer_start 0.25 \
    --epsilon_start 0.3 \
    --world_model "meta-llama/Meta-Llama-3-8B" \
    --mode "train" > "./logs/step_${step}/output.txt" 2>&1 &
