CHECKPOINT_NAME="sdxl/sdxl_cond399_8node_lr5e-7_1step_diffusion1000_gan5e-3_guidance8_noinit_noode_checkpoint_model_024000"
OUTPUT_PATH="model/sdxl"

mkdir $OUTPUT_PATH
# wget  https://huggingface.co/tianweiy/DMD2/resolve/main/model/$CHECKPOINT_NAME/optimizer.bin?download=true -O $OUTPUT_PATH/optimizer.bin
# wget  https://huggingface.co/tianweiy/DMD2/resolve/main/model/$CHECKPOINT_NAME/optimizer_1.bin?download=true -O $OUTPUT_PATH/optimizer_1.bin
# wget  https://hf-mirror.com/tianweiy/DMD2/resolve/main/model/$CHECKPOINT_NAME/pytorch_model.bin?download=true -O $OUTPUT_PATH/pytorch_model.bin
wget  https://hf-mirror.com/tianweiy/DMD2/resolve/main/model/$CHECKPOINT_NAME/pytorch_model_1.bin?download=true -O $OUTPUT_PATH/pytorch_model_1.bin
# wget  https://huggingface.co/tianweiy/DMD2/resolve/main/model/$CHECKPOINT_NAME/scheduler.bin?download=true -O $OUTPUT_PATH/scheduler.bin
# wget  https://huggingface.co/tianweiy/DMD2/resolve/main/model/$CHECKPOINT_NAME/scheduler_1.bin?download=true -O $OUTPUT_PATH/scheduler_1.bin
# wget  https://huggingface.co/tianweiy/DMD2/resolve/main/model/$CHECKPOINT_NAME/random_states_0.pkl?download=true -O $OUTPUT_PATH/random_states_0.pkl
