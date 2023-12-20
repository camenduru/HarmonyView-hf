from functools import partial

from PIL import Image
import numpy as np
import gradio as gr
import torch
import os
import fire
from omegaconf import OmegaConf

from ldm.models.diffusion.sync_dreamer import SyncDDIMSampler, SyncMultiviewDiffusion
from ldm.util import add_margin, instantiate_from_config
from sam_utils import sam_init, sam_out_nosave

import torch
_TITLE = '''SyncDreamer: Generating Multiview-consistent Images from a Single-view Image'''
_DESCRIPTION = '''
<div>
<a style="display:inline-block" href="https://liuyuan-pal.github.io/SyncDreamer/"><img src="https://img.shields.io/badge/SyncDremer-Homepage-blue"></a>
<a style="display:inline-block; margin-left: .5em" href="https://arxiv.org/abs/2309.03453"><img src="https://img.shields.io/badge/2309.03453-f9f7f7?logo=data:image/png;base64,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"></a>
<a style="display:inline-block; margin-left: .5em" href='https://github.com/liuyuan-pal/SyncDreamer'><img src='https://img.shields.io/github/stars/liuyuan-pal/SyncDreamer?style=social' /></a>
</div>
Given a single-view image, SyncDreamer is able to generate multiview-consistent images, which enables direct 3D reconstruction with NeuS or NeRF without SDS loss </br>

Procedure: </br>
**Step 1**. Upload an image or select an example.  ==> The foreground is masked out by SAM and we crop it as inputs. </br>
**Step 2**. Select "Elevation angle "and click "Run generation". ==> Generate multiview images. The **Elevation angle** is the elevation of the input image. (This costs about 30s.) </br>
You may adjust the **Crop size** and **Elevation angle** to get a better result! <br>
To reconstruct a NeRF or a 3D mesh from the generated images, please refer to our [github repository](https://github.com/liuyuan-pal/SyncDreamer). <br>
We have heavily borrowed codes from [One-2-3-45](https://huggingface.co/spaces/One-2-3-45/One-2-3-45), which is also an amazing single-view reconstruction method. 
'''
_USER_GUIDE0 = "Step1: Please upload an image in the block above (or choose an example shown in the left)."
# _USER_GUIDE1 = "Step1: Please select a **Crop size** and click **Crop it**."
_USER_GUIDE2 = "Step2: Please choose a **Elevation angle** and click **Run Generate**. The **Elevation angle** is the elevation of the input image. This costs about 30s."
_USER_GUIDE3 = "Generated multiview images are shown below! (You may adjust the **Crop size** and **Elevation angle** to get a better result!)"

others = '''**Step 1**. Select "Crop size" and click "Crop it". ==> The foreground object is centered and resized. </br>'''

deployed = True

if deployed:
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")


class BackgroundRemoval:
    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = self.interface([image])[0]
        return image

def resize_inputs(image_input, crop_size):
    if image_input is None: return None
    alpha_np = np.asarray(image_input)[:, :, 3]
    coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
    min_x, min_y = np.min(coords, 0)
    max_x, max_y = np.max(coords, 0)
    ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
    h, w = ref_img_.height, ref_img_.width
    scale = crop_size / max(h, w)
    h_, w_ = int(scale * h), int(scale * w)
    ref_img_ = ref_img_.resize((w_, h_), resample=Image.BICUBIC)
    results = add_margin(ref_img_, size=256)
    return results

def generate(model, sample_steps, batch_view_num, sample_num, cfg_scale, seed, image_input, elevation_input):
    if deployed:
        assert isinstance(model, SyncMultiviewDiffusion)
        seed=int(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)

        # prepare data
        image_input = np.asarray(image_input)
        image_input = image_input.astype(np.float32) / 255.0
        alpha_values = image_input[:,:, 3:]
        image_input[:, :, :3] = alpha_values * image_input[:,:, :3] + 1 - alpha_values # white background
        image_input = image_input[:, :, :3] * 2.0 - 1.0
        image_input = torch.from_numpy(image_input.astype(np.float32))
        elevation_input = torch.from_numpy(np.asarray([np.deg2rad(elevation_input)], np.float32))
        data = {"input_image": image_input, "input_elevation": elevation_input}
        for k, v in data.items():
            if deployed:
                data[k] = v.unsqueeze(0).cuda()
            else:
                data[k] = v.unsqueeze(0)
            data[k] = torch.repeat_interleave(data[k], sample_num, dim=0)

        if deployed:
            sampler = SyncDDIMSampler(model, sample_steps)
            x_sample = model.sample(sampler, data, cfg_scale, batch_view_num)
        else:
            x_sample = torch.zeros(sample_num, 16, 3, 256, 256)

        B, N, _, H, W = x_sample.shape
        x_sample = (torch.clamp(x_sample,max=1.0,min=-1.0) + 1) * 0.5
        x_sample = x_sample.permute(0,1,3,4,2).cpu().numpy() * 255
        x_sample = x_sample.astype(np.uint8)

        results = []
        for bi in range(B):
            results.append(np.concatenate([x_sample[bi,ni] for ni in range(N)], 1))
        results = np.concatenate(results, 0)
        return Image.fromarray(results)
    else:
        return Image.fromarray(np.zeros([sample_num*256,16*256,3],np.uint8))


def sam_predict(predictor, removal, raw_im):
    if raw_im is None: return None
    if deployed:
        raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
        image_nobg = removal(raw_im.convert('RGB'))
        arr = np.asarray(image_nobg)[:, :, -1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        # image_nobg.save('./nobg.png')

        image_nobg.thumbnail([512, 512], Image.Resampling.LANCZOS)
        image_sam = sam_out_nosave(predictor, image_nobg.convert("RGB"), (x_min, y_min, x_max, y_max))

        # imsave('./mask.png', np.asarray(image_sam)[:,:,3]*255)
        image_sam = np.asarray(image_sam, np.float32) / 255
        out_mask = image_sam[:, :, 3:]
        out_rgb = image_sam[:, :, :3] * out_mask + 1 - out_mask
        out_img = (np.concatenate([out_rgb, out_mask], 2) * 255).astype(np.uint8)

        image_sam = Image.fromarray(out_img, mode='RGBA')
        # image_sam.save('./output.png')
        torch.cuda.empty_cache()
        return image_sam
    else:
        return raw_im

def run_demo():
    # device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    # models = None # init_model(device, os.path.join(code_dir, ckpt))
    cfg = 'configs/syncdreamer.yaml'
    ckpt = 'ckpt/syncdreamer-pretrain.ckpt'
    config = OmegaConf.load(cfg)
    # model = None
    if deployed:
        model = instantiate_from_config(config.model)
        print(f'loading model from {ckpt} ...')
        ckpt = torch.load(ckpt,map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        model = model.cuda().eval()
        del ckpt
        mask_predictor = sam_init()
        removal = BackgroundRemoval()
    else:
        model = None
        mask_predictor = None
        removal = None

    # NOTE: Examples must match inputs
    examples_full = [
        ['hf_demo/examples/monkey.png',30,200],
        ['hf_demo/examples/cat.png',30,200],
        ['hf_demo/examples/crab.png',30,200],
        ['hf_demo/examples/elephant.png',30,200],
        ['hf_demo/examples/flower.png',0,200],
        ['hf_demo/examples/forest.png',30,200],
        ['hf_demo/examples/teapot.png',20,200],
        ['hf_demo/examples/basket.png',30,200],
    ]

    image_block = gr.Image(type='pil', image_mode='RGBA', height=256, label='Input image', tool=None, interactive=True)
    elevation = gr.Slider(-10, 40, 30, step=5, label='Elevation angle of the input image', interactive=True)
    crop_size = gr.Slider(120, 240, 200, step=10, label='Crop size', interactive=True)

    # Compose demo layout & data flow.
    with gr.Blocks(title=_TITLE, css="hf_demo/style.css") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
            # with gr.Column(scale=0):
            #     gr.DuplicateButton(value='Duplicate Space for private use', elem_id='duplicate-button')
        gr.Markdown(_DESCRIPTION)

        with gr.Row(variant='panel'):
            with gr.Column(scale=1.2):
                gr.Examples(
                    examples=examples_full,  # NOTE: elements must match inputs list!
                    inputs=[image_block, elevation, crop_size],
                    outputs=[image_block, elevation, crop_size],
                    cache_examples=False,
                    label='Examples (click one of the images below to start)',
                    examples_per_page=5,
                )

            with gr.Column(scale=0.8):
                image_block.render()
                guide_text = gr.Markdown(_USER_GUIDE0, visible=True)
                fig0 = gr.Image(value=Image.open('assets/crop_size.jpg'), type='pil', image_mode='RGB', height=256, show_label=False, tool=None, interactive=False)


            with gr.Column(scale=0.8):
                sam_block = gr.Image(type='pil', image_mode='RGBA', label="SAM output", height=256, interactive=False)
                crop_size.render()
                # crop_btn = gr.Button('Crop it', variant='primary', interactive=True)
                fig1 = gr.Image(value=Image.open('assets/elevation.jpg'), type='pil', image_mode='RGB', height=256, show_label=False, tool=None, interactive=False)

            with gr.Column(scale=0.8):
                input_block = gr.Image(type='pil', image_mode='RGBA', label="Input to SyncDreamer", height=256, interactive=False)
                elevation.render()
                with gr.Accordion('Advanced options', open=False):
                    cfg_scale = gr.Slider(1.0, 5.0, 2.0, step=0.1, label='Classifier free guidance', interactive=True)
                    sample_num = gr.Slider(1, 2, 1, step=1, label='Sample num', interactive=False, info='How many instance (16 images per instance)')
                    sample_steps = gr.Slider(10, 300, 50, step=10, label='Sample steps', interactive=False)
                    batch_view_num = gr.Slider(1, 16, 16, step=1, label='Batch num', interactive=True)
                    seed = gr.Number(6033, label='Random seed', interactive=True)
                run_btn = gr.Button('Run generation', variant='primary', interactive=True)


        output_block = gr.Image(type='pil', image_mode='RGB', label="Outputs of SyncDreamer", height=256, interactive=False)

        def update_guide2(text, im):
            if im is None:
                return _USER_GUIDE0
            else:
                return text
        update_guide = lambda GUIDE_TEXT: gr.update(value=GUIDE_TEXT)

        image_block.clear(fn=partial(update_guide, _USER_GUIDE0), outputs=[guide_text], queue=False)
        image_block.change(fn=partial(sam_predict, mask_predictor, removal), inputs=[image_block], outputs=[sam_block], queue=True) \
                   .success(fn=resize_inputs, inputs=[sam_block, crop_size], outputs=[input_block], queue=True)\
                   .success(fn=partial(update_guide2, _USER_GUIDE2), inputs=[image_block], outputs=[guide_text], queue=False)\

        crop_size.change(fn=resize_inputs, inputs=[sam_block, crop_size], outputs=[input_block], queue=True)\
                 .success(fn=partial(update_guide, _USER_GUIDE2), outputs=[guide_text], queue=False)
        # crop_btn.click(fn=resize_inputs, inputs=[sam_block, crop_size], outputs=[input_block], queue=False)\
        #                .success(fn=partial(update_guide, _USER_GUIDE2), outputs=[guide_text], queue=False)

        run_btn.click(partial(generate, model), inputs=[sample_steps, batch_view_num, sample_num, cfg_scale, seed, input_block, elevation], outputs=[output_block], queue=True)\
               .success(fn=partial(update_guide, _USER_GUIDE3), outputs=[guide_text], queue=False)

    demo.queue().launch(share=False, max_threads=80)  # auth=("admin", os.environ['PASSWD'])

if __name__=="__main__":
    fire.Fire(run_demo)