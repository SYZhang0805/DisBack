export CHECKPOINT_PATH="model/sdxl"
export WANDB_ENTITY="z2862909626-zhejiang-university"
export WANDB_PROJECT="DMD2"
export FSDP_DIR="fsdp_configs/sdxl"
export RANK=0

# accelerate launch --config_file fsdp_configs/fsdp_1node_debug.yaml main/train_sd.py  \
# accelerate launch --config_file $FSDP_DIR/config_rank$RANK.yaml main/train_sd.py  \
# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file $FSDP_DIR/config_rank$RANK.yaml main/train_sd.py  \
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node=1 --rdzv_id=2345 main/degradation.py \
    --generator_lr 5e-7  \
    --guidance_lr 5e-7 \
    --train_iters 25000 \
    --output_path log/output \
    --cache_dir log/cache \
    --log_path log/log \
    --batch_size 1 \
    --grid_size 1 \
    --initialie_generator --log_iters 100 \
    --resolution 1024 \
    --latent_resolution 128 \
    --seed 10 \
    --real_guidance_scale 8 \
    --fake_guidance_scale 1.0 \
    --max_grad_norm 10.0 \
    --model_id "/hy-tmp/stable-diffusion-xl-base-1.0" \
    --wandb_iters 100 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_name "sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode"  \
    --log_loss \
    --dfake_gen_update_ratio 5 \
    --sdxl \
    --use_fp16 \
    --max_step_percent 0.98 \
    --cls_on_clean_image \
    --gen_cls_loss \
    --gen_cls_loss_weight 5e-3 \
    --guidance_cls_loss_weight 1e-2 \
    --diffusion_gan \
    --diffusion_gan_max_timestep 1000 \
    --conditioning_timestep 399 \
    --train_prompt_path $CHECKPOINT_PATH/captions_laion_score6.25.pkl \
    --real_image_path $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k_lmdb \
    --ckpt_only_path $CHECKPOINT_PATH/sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode_checkpoint_model_024000 \
    # --generator_lora \
    # --lora_rank 8
    #     --fsdp \
