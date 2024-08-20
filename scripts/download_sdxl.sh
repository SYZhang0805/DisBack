CHECKPOINT_PATH="test_download"

# training prompts 
wget  https://huggingface.co/tianweiy/DMD2/resolve/main/data/laion/captions_laion_score6.25.pkl?download=true -O $CHECKPOINT_PATH/captions_laion_score6.25.pkl

# evaluation prompts
wget  https://huggingface.co/tianweiy/DMD2/resolve/main/data/coco/captions_coco14_test.pkl?download=true -O $CHECKPOINT_PATH/captions_coco14_test.pkl

# pre-trained dmd2 model
wget  https://huggingface.co/tianweiy/DMD2/resolve/main/model/sdxl/sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode_checkpoint_model_024000/pytorch_model.bin?download=true -O $CHECKPOINT_PATH/sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode_checkpoint_model_024000/pytorch_model.bin
wget  https://huggingface.co/tianweiy/DMD2/resolve/main/model/sdxl/sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode_checkpoint_model_024000/pytorch_model_1.bin?download=true -O $CHECKPOINT_PATH/sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode_checkpoint_model_024000/pytorch_model_1.bin


mkdir $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k
# real dataset 
for INDEX in {0..59}
do
    # Format the index to be zero-padded to three digits
    INDEX_PADDED=$(printf "%03d" $INDEX)

    # Download the file
    wget "https://huggingface.co/tianweiy/DMD2/resolve/main/data/laion_vae_latents/sdxl_vae_latents_laion_500k/vae_latents_${INDEX_PADDED}.pt?download=true" -O "${CHECKPOINT_PATH}/sdxl_vae_latents_laion_500k/vae_latents_${INDEX_PADDED}.pt"
done

# generate the lmdb database from the downloaded files
python main/data/create_lmdb_iterative.py   --data_path $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k/  --lmdb_path $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k_lmdb

# evaluation images 
wget https://huggingface.co/tianweiy/DMD2/resolve/main/data/coco/coco10k.zip?download=true -O $CHECKPOINT_PATH/coco10k.zip
unzip $CHECKPOINT_PATH/coco10k.zip -d $CHECKPOINT_PATH