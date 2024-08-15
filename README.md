# DisBack
## The official implementation of [Distribution Backtracking Distillation for One-step Diffusion Models](https://github.com/SYZhang0805/DisBack)
Shengyuan Zhang, Ling Yang, Zejian Li*, Chenye Meng, An Zhao, Changyuan Yang, Guang Yang, Zhiyuan Yang, Lingyun Sun

## Abastract
Accelerating the sampling speed of diffusion models remains a significant challenge. Recent score distillation works try to distill a pre-trained multi-step diffusion model into a one-step generator. 
However, existing methods mainly focus on using the endpoint of pre-trained diffusion models as teacher models, overlooking the importance of the convergence trajectory between the one-step generator and the teacher model, which leads to a degradation in distillation performance.
To address this issue, we extend the score distillation process with the entire convergence trajectory of teacher models and propose **Dis**tribution **Back**tracking Distillation (**DisBack**) for distilling one-step generators. DisBask is composed of two stages: _Degradation Recording_ and _Distribution Backtracking_. 
_Degradation Recording_ is designed for obtaining the convergence trajectory of teacher models, which obtains the degradation path from the trained teacher model to the untrained initial model. 
The degradation path implicitly represents the intermediate distributions of teacher models.
Then _Distribution Backtracking_ trains a student generator to backtrack the intermediate distributions for approximating the convergence trajectory of teacher models.
Extensive experiments show that the DisBack achieves faster and better convergence than the existing distillation method and accomplishes comparable generation performance.
Notably, DisBack is easy to implement and can be generalized to existing distillation methods to boost performance.

## One-step text-to-image generation
![](https://github.com/SYZhang0805/DisBack/blob/main/samples/samples1.png)

## Inference
### Use the distilled SDXL model to do the one-step text-to-image generation
```python
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "SYZhang0805/DisBack"
ckpt_name = "SDXL_DisBack.bin"

unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cuda"))

pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
prompt="A photo of a dog." 
image=pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, timesteps=[399], height=1024, width=1024).images[0]
image.save('output.png', 'PNG')
```

## Use DisBack
DisBack can be applied to the original score distillation process using the following pseudocode.

```python
Distill a one-step generator by DisBack.

Input:
    - Initial generator G^0_{stu}
    - Pre-trained diffusion model s_θ

Output:
    - One-step generator G^*_{stu}

Step 1: Degradation Recording
---------------------------------
1. Initialize s^\prime_θ ← s_θ
2. While not converge:
    a. Generate samples x_0 using G^0_{stu}
    b. Compute noisy samples x_t = x_0 + σ_t * ε
    c. Update model parameters θ by score matching.
    d. Save intermediate checkpoints s^\prime_{θ_i}
  Get the degradation path {s^\prime_{θ_i} | i = 0, ..., N}
  Reverse the degradation path and get the convergence trajectory

Step 2: Distribution Backtracking
---------------------------------
1. Initialize s_φ ← s^\prime_{θ_N}
2. Distileld the intermediate checkpoints s^\prime_{θ_i} in the convergence trajectory sequentially
   For i from N-1 to 0:
    a. While not converge:
        i. Update generator using VSD loss:
            E_{t,ε} [ s_φ(x_t, t) - s′_θi(x_t, t) ] ∂x_t/∂η
        ii. Update s_φ using score matching with generated samples:
            ∂/∂φ [ E_{t,ε} || s_φ(x_t, t) - (x_0 - x_t) / σ_t^2 ||_2^2 ]

3. Return the one-step generator G^*_{stu}
```
