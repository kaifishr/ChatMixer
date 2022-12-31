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

Start chatting with the mixer. The model generates text based on the prompt that can be entered via the console.

```console
cd ChatMixer 
python chat.py 
```

Let's see how the mixer performs:

```console
Please enter a prompt.

Why is there something rather than nothing?

The most in my think it's not they went to do you think about the think it's very with
experience the for many of discover the bigger or complex. The most perfect. And I have a 
lot of the most paints of the talking that the self situation of the way. It's the keep 
leave the had the control interesting. Yeah. StarCh, I mean, it's just when you cand the 
into the first, the beautiful mean and cortainly because there's a lot of the same to any 
of course, we could say, it's very simply by the first 
```
 
Well, that looks pretty mixed up.


## TODOs

- Add additional metrics.


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