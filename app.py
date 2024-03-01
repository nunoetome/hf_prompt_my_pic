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



#def greet(name):
#   return "Hello " + name + "!"

#demo = gr.Interface(fn=greet, inputs="text", outputs="text")

with gr.Blocks() as demo:

    gr.Markdown(
    """
    # Prompt My Pic
    This demo uses the IP Adapter FaceID to generate a new image from a prompt.
    More information here: https://huggingface.co/h94/IP-Adapter-FaceID
    """
    )
    with gr.Row():
        gr.Markdown("""
                    This simple demo is way more efficient if is run in a GPU environment where it can run in a few seconds.
                    In a CPU environment it can take a many minutes to run.
                    """)
    with gr.Row():    
        gr.Markdown(
        """
        Buy me a coffee: https://www.buymeacoffee.com/nuno.tome            
        """
        )
    with gr.Row():
        with gr.Column():
            demo_inputs = []
            demo_inputs.append(gr.Textbox(label='text prompt', value='Linkedin profile picture'))
            demo_inputs.append(gr.Image(type='filepath', label='image prompt'))
            with gr.Accordion(label='Advanced options', open=False):
                demo_inputs.append(gr.Textbox(label='negative text prompt', value="monochrome, lowres, bad anatomy, worst quality, low quality, blurry"))
                demo_inputs.append(gr.Slider(maximum=1, minimum=0, value=0.5, step=0.05, label='image prompt scale'))
            btn = gr.Button("Generate")
            
        with gr.Column():
            demo_outputs = []
            gr.Markdown("output: ")
    with gr.Row():
        gr.Markdown("optiosn: ")
    with gr.Row():
        gr.Markdown("exemples: ")    

if __name__ == "__main__":
    demo.launch()