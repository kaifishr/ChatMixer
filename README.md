# **ChatMixer**

A [MLP-Mixer](https://arxiv.org/abs/2105.01601) architecture for character-level natural language processing.


# Installation

To run *ChatMixer*, install the latest master directly from GitHub. For a basic install, run:

```console
git clone https://github.com/kaifishr/ChatMixer
cd ChatMixer 
pip3 install -r requirements.txt
```


# Getting Started

Start training a model by running:

```console
cd ChatMixer 
python train.py 
```

Track important metrics and visualization of embeddings and weight matrices with Tensorboard:

```console
cd ChatMixer 
tensorboard --logdir runs/
```

Start chatting with a Mixer:

```console
cd ChatMixer 
python chat.py 
```

## TODOs

- Add perplexity metric.
- Drop characters / entire words of the input sequence.