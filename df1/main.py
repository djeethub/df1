from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import os, torch

class Df1:
  pipeline = None
  i2i_pipe = None

  scheduler_name = "Euler a"
  scheduler_type = "euler-ancestral"
  model_name = None
  config_save = None
  layers_save = None
  device = None
  torch_dtype = None

  prompt = ""
  negative_prompt = ""
  steps = 30
  width = 512
  height = 512
  guidance_scale = 7.5
  seed = 0


  def __init__(self, model_path, **kwargs):
    self.torch_dtype = kwargs.pop("torch_dtype", None)
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.pipeline = StableDiffusionPipeline.from_ckpt(model_path, scheduler_type=self.scheduler_type, torch_dtype=self.torch_dtype, load_safety_checker=False)
    self.i2i_pipe = StableDiffusionImg2ImgPipeline(vae=self.pipeline.vae, text_encoder=self.pipeline.text_encoder, tokenizer=self.pipeline.tokenizer, unet=self.pipeline.unet, scheduler=self.pipeline.scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False)

    self.config_save = self.pipeline.scheduler.config.copy()
    self.layers_save = self.pipeline.text_encoder.text_model.encoder.layers
    self.model_name, _ = os.path.splitext(os.path.basename(model_path))

    self.pp_pipe()

  def pp_pipe(self):
    self.pipeline.to(self.device)
    self.i2i_pipe.to(self.device)
    if self.device == "cuda":
      try:
        self.pipeline.enable_xformers_memory_efficient_attention()
        self.i2i_pipe.enable_xformers_memory_efficient_attention()
      except Exception as e:
        print(e)
  
  def load_vae(self, model_path):
    vae_repo_id = model_path

    variant = None
    if '.' in vae_repo_id:
      vae_repo_id, variant = vae_repo_id.split('.', 1)

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_repo_id, torch_dtype=self.torch_dtype, variant=variant)

    if vae is not None:
      self.pipeline.vae = vae
      self.i2i_pipe.vae = vae
      self.pp_pipe()

  def set_params(self, **kwargs):
    self.prompt = kwargs.pop("prompt", self.prompt)
    self.negative_prompt = kwargs.pop("negative_prompt", self.negative_prompt)
    self.steps = kwargs.pop("num_inference_steps", self.steps)
    self.width = kwargs.pop("width", self.width)
    self.height = kwargs.pop("height", self.height)
    self.guidance_scale = kwargs.pop("guidance_scale", self.guidance_scale)
    self.seed = kwargs.pop("seed", self.seed)

  def txt2img(self, **kwargs):
    self.set_params(kwargs)

    from datetime import datetime

