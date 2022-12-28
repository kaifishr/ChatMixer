# **ChatMixer**

An [MLP-Mixer](https://arxiv.org/abs/2105.01601) architecture for character-level natural language processing.


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

Start chatting with a mixer:

```console
cd ChatMixer 
python chat.py 
```

## TODOs

- Add perplexity metric.
- Drop characters / entire words of the input sequence.

# References

[MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)


# Citation

If you find this project useful, please use BibTeX to cite it as:

```bibtex
@article{fischer2022chatmixer,
  title   = "ChatMixer",
  author  = "Fischer, Kai",
  journal = "GitHub repository",
  year    = "2022",
  month   = "December",
  url     = "https://github.com/kaifishr/ChatMixer"
}
```


# License

MIT