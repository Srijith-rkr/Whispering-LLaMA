# Whispering-LLaMA: Integrate Whisper Encoder to LLaMA Decoder

- Accepted at **EMNLP 2023 (Main Track)** | <a href='https://arxiv.org/abs/2310.06434'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> | 
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
- ASR Generative Error Correction by leveraging foundational Audio (Whisper) and Langugae (LLaMA) models.
- Fusing Whisper Encoder and LLaMA decoder

<p align="center">  <img src="https://github.com/Srijith-rkr/Whispering-LLaMA/blob/main/images/wl-arch.png" height ="450"> </p>

# Introduction 
We introduce a novel cross-modal fusion technique designed for generative error correction for Automatic Speech Recognition. In an oversimplified sense, We leverage In-Context learning to feed the n-best hypothesis produced by an Acoustic model into a Large Language model and prompt it to predict the most accurate sentence, as shown below.
<p align="center">  <img src="https://github.com/Srijith-rkr/WHISPERing-LLaMA/blob/main/images/Prompt%20overview.svg" height ="450"> </p>

We propose a novel mechanism to fuse the acoustic features from the audio input into the LLM to significantly enhance the performance (28.83\% -> 37.66\% WERR) by leveraging an Audio Foundational model as a feature extractor. We further design our system in a parameter-efficient manner with only 7.97M trainable parameters as shown below. Please refer to the paper [YET] for further information.

<p align="center">  <img src="https://github.com/Srijith-rkr/Whispering-LLaMA/blob/main/images/Adapter_mechanism.svg" width="700"> </p>


# Setup
Clone the repo

```bash
git clone https://github.com/Srijith-rkr/Whispering-LLaMA
cd WHISPERing-LLaMA
```

And use the environment.yml file to install dependencies with Anaconda.

```bash
conda env create -f environment.yml
```

- To obtain the pre-trained Alpaca weights, please refer [here](https://github.com/tatsu-lab/stanford_alpaca#recovering-alpaca-weights). You can then use convert_hf_checkpoint.py to rename the state_dict the [lit-llama](https://github.com/Lightning-AI/lit-llama) implementation
- Hosted Alpaca weights for easy implementation coming soon


You are all set! 🎉

&nbsp;

# Dataset 
We have uploaded our N-best Hypotheses dataset generated using Whisper-Tiny on [Hugging Face PeacefulData](https://huggingface.co/datasets/PeacefulData/HyPoradise-v1-GigaSpeech). The hypotheses were generated using the Hugging Face [GigaSpeech dataset](https://huggingface.co/datasets/speechcolab/gigaspeech) M subset. You will be able to map the hypothesis on our dataset with the audio clips from the Gigaspeeh dataset using the 'ID' tag.

# Model Weights
We are working on hosting the model weights in [Huggin Face/Whispering-LLaMA](https://huggingface.co/Srijith-rkr/Whispering-LLaMA) for easier setup.

# Training & Inference
Please refer to :
- data_preparation to generate your custom n-best hypothesis dataset
- training/WL-M.py to train the best our best model on your dataset
- Inference/WL-M.py to run inference

# Acknowledgements

This implementation builds on 
-  [lit-llama](https://github.com/Lightning-AI/lit-llama) for the Training pipeline.
- [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) for the pre-trained instruction following Language model.
- [Whisper](https://github.com/openai/whisper) to obtain acoustic embeddings.

- ### Reference

If you consider this work would be related or useful for your research, please consider to cite this paper. Thank you!

```bib
@inproceedings{radhakrishnan2023whispering,
  title={Whispering LLaMA: A Cross-Modal Generative Error Correction Framework for Speech Recognition},
  author={Srijith Radhakrishnan, Chao-Han Huck Yang, Sumeer Ahmad Khan, Rohit Kumar, Narsis A. Kiani, David Gomez-Cabrero, Jesper N. Tegner},
  booktitle={Proc. of EMNLP},
  year={2023}
}
```

