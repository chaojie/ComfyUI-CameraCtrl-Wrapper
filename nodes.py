import argparse
import json
import os

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)

import sys
cameractrl_path=f'{comfy_path}/custom_nodes/ComfyUI-CameraCtrl-Wrapper'
sys.path.insert(0,cameractrl_path)

import numpy as np
import torch
from tqdm import tqdm
from packaging import version as pver
from einops import rearrange
from safetensors import safe_open

from omegaconf import OmegaConf
from diffusers import (
    AutoencoderKL,
    DDIMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_vae_checkpoint, \
    convert_ldm_clip_checkpoint

from cameractrl.utils.util import save_videos_grid
from cameractrl.models.unet import UNet3DConditionModelPoseCond
from cameractrl.models.pose_adaptor import CameraPoseEncoder
from cameractrl.pipelines.pipeline_animation import CameraCtrlPipeline
from cameractrl.utils.convert_from_ckpt import convert_ldm_unet_checkpoint
from .camera_utils import *

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[0:4]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[6:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def ray_condition(K, c2w, H, W, device):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def load_personalized_base_model(pipeline, personalized_base_model):
    print(f'Load civitai base model from {personalized_base_model}')
    if personalized_base_model.endswith(".safetensors"):
        dreambooth_state_dict = {}
        with safe_open(personalized_base_model, framework="pt", device="cpu") as f:
            for key in f.keys():
                dreambooth_state_dict[key] = f.get_tensor(key)
    elif personalized_base_model.endswith(".ckpt"):
        dreambooth_state_dict = torch.load(personalized_base_model, map_location="cpu")

    # 1. vae
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, pipeline.vae.config)
    pipeline.vae.load_state_dict(converted_vae_checkpoint)
    # 2. unet
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, pipeline.unet.config)
    _, unetu = pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
    assert len(unetu) == 0
    # 3. text_model
    pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict, text_encoder=pipeline.text_encoder)
    del dreambooth_state_dict
    return pipeline


def get_pipeline(vae,tokenizer,text_encoder,unet, image_lora_rank, image_lora_ckpt, unet_additional_kwargs,
                 unet_mm_ckpt, pose_encoder_kwargs, attention_processor_kwargs,
                 noise_scheduler_kwargs, pose_adaptor_ckpt, personalized_base_model, gpu_id):
    #vae = AutoencoderKL.from_pretrained(ori_model_path, subfolder="vae")
    #tokenizer = CLIPTokenizer.from_pretrained(ori_model_path, subfolder="tokenizer")
    #text_encoder = CLIPTextModel.from_pretrained(ori_model_path, subfolder="text_encoder")
    #unet = UNet3DConditionModelPoseCond.from_pretrained_2d(ori_model_path, subfolder=unet_subfolder,unet_additional_kwargs=unet_additional_kwargs)
    '''
    unet=UNet3DConditionModelPoseCond(unet)
    unet.down_block_types = [
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "DownBlock3D"
    ]
    unet.up_block_types = [
        "UpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D"
    ]
    '''

    pose_encoder = CameraPoseEncoder(**pose_encoder_kwargs)
    print(f"Setting the attention processors")
    unet.set_all_attn_processor(add_spatial_lora=image_lora_ckpt is not None,
                                add_motion_lora=False,
                                lora_kwargs={"lora_rank": image_lora_rank, "lora_scale": 1.0},
                                motion_lora_kwargs={"lora_rank": -1, "lora_scale": 1.0},
                                **attention_processor_kwargs)

    if image_lora_ckpt is not None:
        print(f"Loading the lora checkpoint from {image_lora_ckpt}")
        lora_checkpoints = torch.load(image_lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        print(f'Loading done')

    if unet_mm_ckpt is not None:
        print(f"Loading the motion module checkpoint from {unet_mm_ckpt}")
        mm_checkpoints = torch.load(unet_mm_ckpt, map_location=unet.device)
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        print("Loading done")

    print(f"Loading pose adaptor")
    pose_adaptor_checkpoint = torch.load(pose_adaptor_ckpt, map_location='cpu')
    pose_encoder_state_dict = pose_adaptor_checkpoint['pose_encoder_state_dict']
    pose_encoder_m, pose_encoder_u = pose_encoder.load_state_dict(pose_encoder_state_dict)
    assert len(pose_encoder_u) == 0 and len(pose_encoder_m) == 0
    attention_processor_state_dict = pose_adaptor_checkpoint['attention_processor_state_dict']
    _, attn_proc_u = unet.load_state_dict(attention_processor_state_dict, strict=False)
    assert len(attn_proc_u) == 0
    print(f"Loading done")

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    vae.to(gpu_id)
    text_encoder.to(gpu_id)
    unet.to(gpu_id)
    pose_encoder.to(gpu_id)
    pipe = CameraCtrlPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        pose_encoder=pose_encoder)
    if personalized_base_model is not None:
        load_personalized_base_model(pipeline=pipe, personalized_base_model=personalized_base_model)
    pipe.enable_vae_slicing()
    pipe = pipe.to(gpu_id)

    return pipe

class CameraCtrlLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sd15_ckpt": (folder_paths.get_filename_list("checkpoints"),),
                "ad_v3_sd15_adapter_ckpt": (folder_paths.get_filename_list("loras"), {"default": "v3_sd15_adapter.ckpt"}),
                "ad_v3_ckpt": (os.listdir(os.path.join(folder_paths.models_dir,"animatediff_models")), {"default": "v3_sd15_mm.ckpt"}),
                "cameractrl_ckpt": (folder_paths.get_filename_list("checkpoints"), {"default": "CameraCtrl.ckpt"}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("CameraCtrlPipeline",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,sd15_ckpt,ad_v3_sd15_adapter_ckpt,ad_v3_ckpt,cameractrl_ckpt):
        sd15_ckpt = folder_paths.get_full_path("checkpoints", sd15_ckpt)
        ad_v3_sd15_adapter_ckpt = folder_paths.get_full_path("loras", ad_v3_sd15_adapter_ckpt)
        ad_v3_ckpt=os.path.join(os.path.join(folder_paths.models_dir,"animatediff_models"),ad_v3_ckpt)
        cameractrl_ckpt = folder_paths.get_full_path("checkpoints", cameractrl_ckpt)
        
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_single_file(sd15_ckpt).to("cuda")
        
        rank = -1
        setup_for_distributed(rank == 0)
        gpu_id = rank % torch.cuda.device_count()

        model_configs = OmegaConf.load(f'{cameractrl_path}/configs/train_cameractrl/adv3_256_384_cameractrl_relora.yaml')
        unet_additional_kwargs = model_configs[
            'unet_additional_kwargs'] if 'unet_additional_kwargs' in model_configs else None
        noise_scheduler_kwargs = model_configs['noise_scheduler_kwargs']
        pose_encoder_kwargs = model_configs['pose_encoder_kwargs']
        attention_processor_kwargs = model_configs['attention_processor_kwargs']

        unet=pipe.unet
        fused_state_dict = unet.state_dict()
        lora_state_dict = torch.load(ad_v3_sd15_adapter_ckpt, map_location='cuda')
        if 'state_dict' in lora_state_dict:
            lora_state_dict = lora_state_dict['state_dict']
        print(f'Loading done')
        print(f'Fusing the lora weight to unet weight')
        used_lora_key = []
        for lora_key in ['to_q', 'to_k', 'to_v', 'to_out']:
            unet_keys = [x for x in fused_state_dict.keys() if lora_key in x and "bias" not in x]
            print(f'There are {len(unet_keys)} unet keys for lora key: {lora_key}')
            for unet_key in unet_keys:
                prefixes = unet_key.split('.')
                idx = prefixes.index(lora_key)
                lora_down_key = ".".join(prefixes[:idx]) + f".processor.{lora_key}_lora.down" + f".{prefixes[-1]}"
                lora_up_key = ".".join(prefixes[:idx]) + f".processor.{lora_key}_lora.up" + f".{prefixes[-1]}"
                assert lora_down_key in lora_state_dict and lora_up_key in lora_state_dict
                print(f'Fusing lora weight for {unet_key}')
                fused_state_dict[unet_key] = fused_state_dict[unet_key] + torch.bmm(lora_state_dict[lora_up_key][None, ...], lora_state_dict[lora_down_key][None, ...])[0] * 1.0
                used_lora_key.append(lora_down_key)
                used_lora_key.append(lora_up_key)
        assert len(set(used_lora_key) - set(lora_state_dict.keys())) == 0
        print(f'Fusing done')
        from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
        save_path = os.path.join(os.path.join(cameractrl_path, 'unet_webvidlora_v3'),SAFETENSORS_WEIGHTS_NAME)
        print(f'Saving the fused state dict to {save_path}')

        from safetensors.torch import save_file
        save_file(fused_state_dict, save_path)
        unet = UNet3DConditionModelPoseCond.from_pretrained_2d(cameractrl_path, subfolder='unet_webvidlora_v3',unet_additional_kwargs=unet_additional_kwargs)

        print(f'Constructing pipeline')
        image_lora_rank=2
        image_lora_ckpt=None
        personalized_base_model=None
        pipeline = get_pipeline(pipe.vae,pipe.tokenizer,pipe.text_encoder,unet, image_lora_rank, image_lora_ckpt,
                                unet_additional_kwargs, ad_v3_ckpt, pose_encoder_kwargs, attention_processor_kwargs,
                                noise_scheduler_kwargs, cameractrl_ckpt,
                                personalized_base_model, f"cuda:{gpu_id}")
        return (pipeline,)

class CameraBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "speed":("FLOAT",{"default":1.0}),
                "video_length":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("CameraPose",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,camera_pose,speed,video_length):
        camera_dict = {
                "motion":[camera_pose],
                "mode": "Basic Camera Poses",  # "First A then B", "Both A and B", "Custom"
                "speed": speed,
                "complex": None
                } 
        motion_list = camera_dict['motion']
        mode = camera_dict['mode']
        speed = camera_dict['speed'] 
        angle = np.array(CAMERA[motion_list[0]]["angle"])
        T = np.array(CAMERA[motion_list[0]]["T"])
        RT = get_camera_motion(angle, T, speed, video_length)
        return (RT,)

class CameraJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose1":("CameraPose",),
                "camera_pose2":("CameraPose",),
            },
        }

    RETURN_TYPES = ("CameraPose",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,camera_pose1,camera_pose2):
        RT = combine_camera_motion(camera_pose1, camera_pose2)
        return (RT,)

class CameraCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose1":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "camera_pose2":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "camera_pose3":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "camera_pose4":(["Static","Pan Up","Pan Down","Pan Left","Pan Right","Zoom In","Zoom Out","ACW","CW"],{"default":"Static"}),
                "speed":("FLOAT",{"default":1.0}),
                "video_length":("INT",{"default":16}),
            },
        }

    RETURN_TYPES = ("CameraPose",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,camera_pose1,camera_pose2,camera_pose3,camera_pose4,speed,video_length):
        angle = np.array(CAMERA[camera_pose1]["angle"]) + np.array(CAMERA[camera_pose2]["angle"]) + np.array(CAMERA[camera_pose3]["angle"]) + np.array(CAMERA[camera_pose4]["angle"])
        T = np.array(CAMERA[camera_pose1]["T"]) + np.array(CAMERA[camera_pose2]["T"]) + np.array(CAMERA[camera_pose3]["T"]) + np.array(CAMERA[camera_pose4]["T"])
        RT = get_camera_motion(angle, T, speed, video_length)
        return (RT,)

class CameraTrajectory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_pose":("CameraPose",),
                "fx":("FLOAT",{"default":0.474812461, "min": 0, "max": 1, "step": 0.000000001}),
                "fy":("FLOAT",{"default":0.844111024, "min": 0, "max": 1, "step": 0.000000001}),
                "cx":("FLOAT",{"default":0.5, "min": 0, "max": 1, "step": 0.01}),
                "cy":("FLOAT",{"default":0.5, "min": 0, "max": 1, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING","INT",)
    RETURN_NAMES = ("camera_trajectory","video_length",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,camera_pose,fx,fy,cx,cy):
        #print(camera_pose)
        camera_pose_list=camera_pose.tolist()
        trajs=[]
        for cp in camera_pose_list:
            traj=[fx,fy,cx,cy,0,0]
            traj.extend(cp[0])
            traj.extend(cp[1])
            traj.extend(cp[2])
            trajs.append(traj)
        print(json.dumps(trajs))
        
        return (json.dumps(trajs),len(trajs),)

class CameraCtrlRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":("CameraCtrlPipeline",),
                "camera_trajectory":("STRING",{"multiline": True, "default":"[[0.474812461,0.844111024,0.5,0.5,0,0,0.780003667,0.059620168,-0.622928321,0.726968666,-0.062449891,0.997897983,0.017311305,0.217967188,0.622651041,0.025398925,0.782087326,-1.002211444],[0.474812461,0.844111024,0.5,0.5,0,0,0.743836701,0.064830206,-0.66520977,0.951841944,-0.068305343,0.997446954,0.020830527,0.206496789,0.664861917,0.029942872,0.746365905,-1.084913992],[0.474812461,0.844111024,0.5,0.5,0,0,0.697046876,0.070604131,-0.713540971,1.208789672,-0.074218854,0.996899366,0.026138915,0.196421447,0.713174045,0.034738146,0.700125754,-1.130142078],[0.474812461,0.844111024,0.5,0.5,0,0,0.635762572,0.077846259,-0.767949164,1.465161122,-0.080595709,0.996158004,0.034256749,0.157107229,0.767665446,0.040114246,0.639594078,-1.13689307],[0.474812461,0.844111024,0.5,0.5,0,0,0.593250692,0.083153486,-0.800711632,1.635091834,-0.085384794,0.995539784,0.040124334,0.135863998,0.800476789,0.04456481,0.597704709,-1.166997229],[0.474812461,0.844111024,0.5,0.5,0,0,0.555486798,0.087166689,-0.826943994,1.803789619,-0.089439675,0.99498421,0.044799786,0.145490422,0.826701283,0.049075913,0.560496747,-1.24382735],[0.474812461,0.844111024,0.5,0.5,0,0,0.523399472,0.09026666,-0.847292721,1.945815368,-0.093254104,0.994468153,0.048340045,0.174777447,0.846969128,0.053712368,0.528921843,-1.336914479],[0.474812461,0.844111024,0.5,0.5,0,0,0.491546303,0.09212707,-0.865964711,2.093852892,-0.095617607,0.994085968,0.051482171,0.196702533,0.865586221,0.057495601,0.497448236,-1.43970938],[0.474812461,0.844111024,0.5,0.5,0,0,0.475284129,0.093297184,-0.87487179,2.200792438,-0.096743606,0.993874133,0.053430639,0.209217395,0.874497354,0.059243519,0.481398523,-1.547068315],[0.474812461,0.844111024,0.5,0.5,0,0,0.464444131,0.093880348,-0.880612373,2.324141986,-0.097857766,0.993716478,0.054326952,0.220651207,0.880179226,0.060942926,0.470712721,-1.712512928],[0.474812461,0.844111024,0.5,0.5,0,0,0.458157241,0.093640216,-0.883925021,2.44310089,-0.098046601,0.993691206,0.054448847,0.257385043,0.883447111,0.061719712,0.464447916,-1.885672329],[0.474812461,0.844111024,0.5,0.5,0,0,0.457354397,0.09350872,-0.884354591,2.543246338,-0.097820736,0.993711591,0.054482624,0.281562244,0.883888066,0.061590351,0.463625461,-2.094829165],[0.474812461,0.844111024,0.5,0.5,0,0,0.465170115,0.093944497,-0.880222261,2.606377358,-0.097235762,0.99375838,0.054675922,0.277376127,0.879864752,0.060155477,0.471401453,-2.299280675],[0.474812461,0.844111024,0.5,0.5,0,0,0.511845231,0.090872414,-0.854257941,2.5767741,-0.093636356,0.994366586,0.049672548,0.270516319,0.853959382,0.054564942,0.517470777,-2.624374352],[0.474812461,0.844111024,0.5,0.5,0,0,0.590568483,0.083218277,-0.802685261,2.398318316,-0.085610889,0.995516419,0.04022257,0.282138215,0.80243355,0.044964414,0.59504503,-3.012309268],[0.474812461,0.844111024,0.5,0.5,0,0,0.684302032,0.072693504,-0.725566208,2.086323553,-0.074529484,0.996780157,0.029575195,0.310959312,0.725379944,0.03383771,0.68751651,-3.456740526]]"}),
                "image_width":("INT",{"default":384}),
                "image_height":("INT",{"default":256}),
                "original_pose_width":("INT",{"default":1280}),
                "original_pose_height":("INT",{"default":720}),
                "prompt":("STRING",{"multiline": True, "default":"A serene mountain lake at sunrise, with mist hovering over the water."}),
                "negative_prompt":("STRING",{"multiline": True, "default":"Strange motion trajectory, a poor composition and deformed video, worst quality, normal quality, low quality, low resolution, duplicate and ugly"}),
                "video_length":("INT",{"default":16}),
                "num_inference_steps":("INT",{"default":30}),
                "guidance_scale":("FLOAT",{"default":6.0}),
                "seed":("INT",{"default":1234}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "CameraCtrl"

    def run(self,pipeline,camera_trajectory,image_width,image_height,original_pose_width,original_pose_height,prompt,negative_prompt,video_length,num_inference_steps,guidance_scale,seed):
        rank = -1
        setup_for_distributed(rank == 0)
        gpu_id = rank % torch.cuda.device_count()

        generator = torch.Generator(device='cuda').manual_seed(seed)
        device = torch.device(f"cuda:{gpu_id}")
        print('Done')
        print('Loading K, R, t matrix')
        #with open(args.trajectory_file, 'r') as f:
        #    poses = f.readlines()
        #poses = [pose.strip().split(' ') for pose in poses[1:]]

        poses=json.loads(camera_trajectory)
        cam_params = [[float(x) for x in pose] for pose in poses]
        cam_params = [Camera(cam_param) for cam_param in cam_params]
        sample_wh_ratio = image_width / image_height
        pose_wh_ratio = original_pose_width / original_pose_height
        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = image_height * pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / image_width
        else:
            resized_ori_h = image_width / pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / image_height
        intrinsic = np.asarray([[cam_param.fx * image_width,
                                cam_param.fy * image_height,
                                cam_param.cx * image_width,
                                cam_param.cy * image_height]
                                for cam_param in cam_params], dtype=np.float32)

        K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
        c2ws = get_relative_pose(cam_params)
        c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
        plucker_embedding = ray_condition(K, c2ws, image_height, image_width, device='cpu')[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
        plucker_embedding = plucker_embedding[None].to(device)  # B V 6 H W
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")
        captions=[prompt]
        negative_prompts=[negative_prompt]
        specific_seeds=[seed]
        n_procs=8
        N = int(len(captions) // n_procs)
        remainder = int(len(captions) % n_procs)
        prompts_per_gpu = [N + 1 if gpu_id < remainder else N for gpu_id in range(n_procs)]
        low_idx = sum(prompts_per_gpu[:gpu_id])
        high_idx = low_idx + prompts_per_gpu[gpu_id]
        prompts = captions[low_idx: high_idx]
        negative_prompts = negative_prompts[low_idx: high_idx] if negative_prompts is not None else None
        specific_seeds = specific_seeds[low_idx: high_idx] if specific_seeds is not None else None
        print(f"rank {rank} / {torch.cuda.device_count()}, number of prompts: {len(prompts)}")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        for local_idx, caption in tqdm(enumerate(prompts)):
            if specific_seeds is not None:
                specific_seed = specific_seeds[local_idx]
                generator.manual_seed(specific_seed)
            videos = pipeline(
                prompt=caption,
                negative_prompt=negative_prompts[local_idx] if negative_prompts is not None else None,
                pose_embedding=plucker_embedding,
                video_length=video_length,
                height=image_height,
                width=image_width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).videos  # [1, 3, f, h, w]
            videos=videos.permute(0,2,3,4,1)
            return videos


NODE_CLASS_MAPPINGS = {
    "CameraCtrlLoader":CameraCtrlLoader,
    "CameraCtrlRun":CameraCtrlRun,
    "CameraBasic":CameraBasic,
    "CameraJoin":CameraJoin,
    "CameraCombine":CameraCombine,
    "CameraTrajectory":CameraTrajectory,
}
