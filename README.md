# **ChatMixer**

An [MLP-Mixer](https://arxiv.org/abs/2105.01601) architecture with experimental [meta layers](#meta-layers) for character-level natural language processing.


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

Let's see how the mixer performs after a short training on a single GPU.

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


# Meta Layers

Meta layers are an experimental feature of this repository. Meta layers do not directly transform an input $x$, but first compute a weight matrix $W$ and a bias vector $b$, based on input $x$. In a subsequent step, the computed weight matrix $W$ and bias vector $b$ are then used for the linear transformation of the incoming data.

For a target transformation $y \in \mathbb{R}^{m \times 1}$ and an input $x \in \mathbb{R}^{n \times 1}$, the bias parameters $b \in \mathbb{R}^{m \times 1}$ are computed as follows

$$b = W_b x$$

with $W_b \in \mathbb{R}^{m \times n}$. 

Computing the weight matrix $W \in \mathbb{R}^{m \times n}$ requires two steps. First we compute the weights of the weight matrix

$$W = W_w x$$

with $W_w \in \mathbb{R}^{mn \times n}$. In a second step we reshape the column parameter matrix computed to derive the final weight matrix:

$$W \in \mathbb{R}^{mn} \rightarrow W \in \mathbb{R}^{m \times n}$$

Finally, we compute a linear transformation to the incoming data $x$ as usual:

$$y = Wx+b$$

As can be easily seen, this type of layer comes with high computational and memory costs.


# TODOs

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