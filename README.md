<div align="center">
  <img src="assets/teaser.png">

<h1>Infinite Worlds with Versatile Interactions</h1>

Robbyant Team

</div>


<div align="center">

[![Page](https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-Demo-00bfff)](https://technology.robbyant.com/lingbot-world-v2)
[![Tech Report](https://img.shields.io/static/v1?label=Paper&message=PDF&color=red&logo=arxiv)](https://arxiv.org/abs/2607.07534)
[![Model](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=yellow)](https://huggingface.co/collections/robbyant/lingbot-world-v2)
[![Model](https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=purple)](https://modelscope.cn/collections/Robbyant/LingBot-World-V2)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-green)](LICENSE.txt)


</div>

-----

We present **LingBot-World 2.0** (also known as **LingBot-World-Infinity**), an advanced iteration of [LingBot-World](https://technology.robbyant.com/lingbot-world) featuring four distinct upgrades.
- **Unbounded Interaction Horizon**: Our model achieves an unbounded interaction horizon while maintaining consistent output quality, benefiting from a carefully crafted causal pretraining paradigm.
- **Rapid Response Time**: Through distilling a real-time variant from the base model, our system guarantees rapid response time, sufficient to drive 720p video streams at 60 fps.
- **Highly Diverse Interactive Elements**: Compared to the previous version, this update introduces highly diverse interactive elements, comprising a broader spectrum of actions (*e.g.*, attacking, archery, spell-casting, and shooting) alongside a richer variety of text-driven events.
- **Agentic Harness**: We pioneer the integration of an agentic harness within the domain of world modeling, wherein a pilot agent is tasked with planning and executing character behaviors, while a director agent is responsible for synthesizing novel environmental elements as the scene progresses.


## 🚀 Try it now
The real-time version of LingBot-World-Infinity is available on two platforms. We thank [Reactor](https://www.reactor.inc/lingbot-world-v2) and [LingGuang](https://www.lingguang.com/support) for their support:
- **International (Web)**: Experience it on [Reactor](https://www.reactor.inc/lingbot-world-v2).
- **Domestic (Mobile)**: Experience it on [LingGuang](https://www.lingguang.com/support).

> **Note:** Reactor and LingGuang provide a convenient way to try LingBot-World-Infinity in real time. In our official setup, the model runs at full capability. To experience our official demo, join us at [WAIC 2026](https://waica2026.worldaic.com.cn/).

## 🎬 Demo Gallery

<div align="center">
  <video src="https://github.com/user-attachments/assets/ab2a81a8-56f7-4328-a5cc-80477151c61c" width="100%" poster=""> </video>
  <video src="https://github.com/user-attachments/assets/2a1a4864-7809-4bff-ab08-32bd30099581" width="100%" poster=""> </video>
  <video src="https://github.com/user-attachments/assets/f1059674-a7e7-45b1-8738-627d811d7bee" width="100%" poster=""> </video>
  <video src="https://github.com/user-attachments/assets/538097aa-6c02-48e1-9802-563416f6191a" width="100%" poster=""> </video>
  <video src="https://github.com/user-attachments/assets/e7e0749a-9ca9-4502-a846-661c41b48096" width="100%" poster=""> </video>
  <video src="https://github.com/user-attachments/assets/09970b6c-990d-4e40-bd8b-82755400fa9d" width="100%" poster=""> </video>
</div>


<p align="center"><i>✨ For more high-fidelity and compelling demos, please visit our <a href="https://technology.robbyant.com/lingbot-world-v2">Project Page</a>.</i></p>

## 🔥 News
- Sep. 10, 2026: 🎉 We release the remaining full model variants: the 14B model’s causal-pretrained and bidirectional variants, and the 1.3B model’s causal-fast variant.
- Jul. 9, 2026: 🎉 We release the technical report, inference code, and models for LingBot-World-Infinity.

## 📋 TODO
- [x] Release the causal-fast inference code and model of the 14B model
- [x] Release the causal-pretrained model of the 14B model
- [x] Release the bidirectional model of the 14B model
- [x] Release the causal-fast model of the 1.3B model

## ⚙️ Quick Start
This codebase is built upon [Wan2.2](https://github.com/Wan-Video/Wan2.2). Please refer to their documentation for installation instructions.
### Installation
Clone the repo:
```sh
git clone https://github.com/robbyant/lingbot-world-v2.git
cd lingbot-world-v2
```
Install dependencies:
```sh
# Ensure torch >= 2.4.0
pip install -r requirements.txt
```
Install [`flash_attn`](https://github.com/Dao-AILab/flash-attention):
```sh
pip install flash-attn --no-build-isolation
```
### Model Download

| Model | Model Type | Model Size | Download Links |
| :---  | :--- | :--- | :--- |
| **lingbot-world-v2-14b-causal-fast** | causal-fast | 14B | 🤗 [HuggingFace](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast) 🤖 [ModelScope](https://www.modelscope.cn/models/Robbyant/lingbot-world-v2-14b-causal-fast) |
| **lingbot-world-v2-14b-causal-pretrain** | causal-pretrain | 14B | 🤗 [HuggingFace](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-pretrain) |
| **lingbot-world-v2-14b-bid** | bidirectional | 14B | 🤗 [HuggingFace](https://huggingface.co/robbyant/lingbot-world-v2-14b-bid) |
| **lingbot-world-v2-1.3b-causal-fast** | causal-fast | 1.3B | 🤗 [HuggingFace](https://huggingface.co/robbyant/lingbot-world-v2-1.3b-causal-fast) |


Download models using huggingface-cli:
```sh
pip install "huggingface_hub[cli]"
huggingface-cli download robbyant/lingbot-world-v2-14b-causal-fast --local-dir ./lingbot-world-v2-14b-causal-fast
```
Download models using modelscope-cli:
 ```sh
pip install modelscope
modelscope download robbyant/lingbot-world-v2-14b-causal-fast --local_dir ./lingbot-world-v2-14b-causal-fast
```

### Inference

We provide `generate.py` for causal inference with KV caching, which processes video frames chunk-by-chunk instead of all at once.
<!-- The `--infer_mode` flag selects the inference mode:

| infer_mode | Model | Sampling |
| :--- | :--- | :--- |
| `causal_fast` (default) | Distilled few-step model (`LingBot-World-Fast`) | 4 steps per chunk, no CFG |
| `causal_pretrain` | Pretrained causal model | 40 steps per chunk with CFG | -->

- `causal_fast` — 480P, multi-GPU:
  ``` sh
  torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 480*832 --ckpt_dir lingbot-world-v2-14b-causal-fast --image examples/03/image.jpg --action_path examples/03 --dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 361 --local_attn_size 18 --sink_size 6 --prompt "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere."
  ```

<!-- - `causal_pretrain` — 480P, multi-GPU:
  ``` sh
  torchrun --nproc_per_node=8 generate.py --task i2v-A14B --infer_mode causal_pretrain --size 480*832 --ckpt_dir lingbot-world-v2-14b-causal-pretrain --image examples/03/image.jpg --action_path examples/03 --dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 81 --prompt "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere."
  ``` -->

You can also use the provided `run_fast.sh` script:
``` sh
bash run_fast.sh <weights_dir> <frame_num>
# e.g. bash run_fast.sh lingbot-world-v2-14b-causal-fast 361
```

### Deployment
We do NOT plan to release our deployment code. If you would like to deploy our model yourself, please refer to the LingBot-World deployment in [SGLang](https://docs.sglang.io/cookbook/diffusion/LingBot-World/LingBot-World-2.0) or [flashdreams](https://github.com/NVIDIA/flashdreams).

## 📚 Related Projects
- [LingBot-World](https://github.com/robbyant/lingbot-world)

## 📜 License
This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). The project is available for non-commercial use only: you may share and adapt it with proper attribution, but derivative works must be distributed under the same license. Please refer to the [LICENSE file](LICENSE.txt) for the full text, including details on rights and restrictions.

## ✨ Acknowledgement
We would like to express our gratitude to the Wan Team for open-sourcing their code and models. Their contributions have been instrumental to the development of this project.

## 📖 Citation
If you find this work useful for your research, please cite our paper:

```
@article{lingbot-world-v2,
      title={Infinite Worlds with Versatile Interactions}, 
      author={Zelin Gao and Qiuyu Wang and Jiapeng Zhu and Jingye Chen and Zichen Liu and Qingyan Bai and Jiahao Wang and Yufeng Yuan and Hanlin Wang and Yichong Lu and Ka Leong Cheng and Haojie Zhang and Jian Gao and Tianrui Feng and Yuzheng Liu and Yao Yao and Yinghao Xu and Xing Zhu and Yujun Shen and Hao Ouyang},
      journal={arXiv preprint arXiv:2607.07534},
      year={2026}
}
```
