# ChatMixer

A PyTorch implementation of MLP-Mixer and ConvMixer architectures with experimental [meta layers](#meta-layers) for character-level natural language processing.


## Installation

To run *ChatMixer*, install the latest master directly from GitHub. For a basic install, run:

```console
git clone https://github.com/kaifishr/ChatMixer
cd ChatMixer 
pip3 install -r requirements.txt
```


## Getting Started

Start training a model by running:

```console
cd ChatMixer 
python train.py 
```

By default, the model is trained on the [Lexicap](https://karpathy.ai/lexicap/) dataset.

Start chatting with the mixer. The model generates text based on the prompt that can be entered via the console.

```console
cd ChatMixer 
python chat.py 
```

Let's see how the mixer performs after training on a single NVIDIA RTX 2080 Ti for one hour.

```console
[User]
Why is there something rather than nothing?


[ChatMixer]
I mean, I think it's a dream standard the beautiful security of your own for the 
real world, and then the presentation of what's the thing that it would be a 
research in the world that come out there. We all the different than the reality 
and so on. Now the right context of the problem is in the war and then the 
philosophical physics has a lot of problems and the community a good for the 
most money and so on. And then you make a completely engaged on the whole people 
are gonna be some people wh
```
 
Well, even thought that looks pretty mixed up, the mixer model is able to learn some english words and something that looks like sentences. Haha.


## Meta Layers

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


## Weight Visualization

Some important metrics and trained parameters of the token-mixing MLP blocks can be visualized with Tensorboard:

```console
cd ChatMixer 
tensorboard --logdir runs/
```

The following visualizations show some of the weights learned during training by the token-mixing MLPs.

<center>

| Layer 1 | Layer 2 | Layer 3  | Layer 4  | Layer 5  | Layer 6  | Layer 7  | Layer 8 |
|---|---|---|---|---|---|---|---|
| ![](/docs/images/layer_01.png) | ![](/docs/images/layer_02.png) | ![](/docs/images/layer_03.png) | ![](/docs/images/layer_04.png) | ![](/docs/images/layer_05.png) | ![](/docs/images/layer_06.png) | ![](/docs/images/layer_07.png) | ![](/docs/images/layer_08.png)

</center>


## TODO:

- Add better encoding scheme to support longer sequences.
- Allow chat to receive command line arguments.
- Add additional metrics.


## References

[1] [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)

[2] [Patches Are All You Need?](https://arxiv.org/abs/2201.09792)



## Citation

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


## License

MIT
