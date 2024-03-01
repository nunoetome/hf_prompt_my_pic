import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import gradio as gr
from huggingface_hub import hf_hub_download
from datetime import datetime



def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")


demo.launch()