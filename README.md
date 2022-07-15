![alt text](https://i.imgur.com/rjw4eFg.png)

[comment]: <> ([![]&#40;https://img.shields.io/badge/State_of_the_Art-Data--Efficient_RL-blue.svg??style=flat&logo=google-analytics&#41;]&#40;&#41;<br>)

[comment]: <> ([![]&#40;https://img.shields.io/badge/Modern,_Academic--Standard-Image_Classification-blue.svg??style=flat&logo=instatus&#41;]&#40;&#41;<br>)

[comment]: <> ([![]&#40;https://img.shields.io/badge/Technically--Working-Generative_Modeling-blue.svg??style=flat&logo=angellist&#41;]&#40;&#41;<br>)

[comment]: <> ([![]&#40;https://img.shields.io/badge/In--Progress-Multi_Modalities-red.svg??style=flat&logo=plex&#41;]&#40;&#41;<br>)

[comment]: <> ([![]&#40;https://img.shields.io/badge/Unified_in_one_Framework-Seamless,_General.svg??style=flat&logo=immer&#41;]&#40;&#41;)

### Quick Links

- [Setup](#wrench-setting-up)

- [Examples](#mag-sample-scripts)

- [Agents and performances](#bar_chart-agents--performances)

# :runner: Running The Code

To start a train session, once [installed](#wrench-setting-up):

```console
python Run.py
```

Defaults:

```Agent=Agents.DQNAgent```

```task=atari/pong```

Plots, logs, generated images, and videos are automatically stored in: ```./Benchmarking```.

![alt text](https://i.imgur.com/2jhOPib.gif)

Welcome ye, weary Traveller.

>Stop here and rest at our local tavern,
>
> Where all your reinforcements and supervisions be served, à la carte!

Drink up! :beers:

# :pen: Paper & Citing

For detailed documentation, [see our :scroll:](https://arxiv.com).

[comment]: <> ([![arXiv]&#40;https://img.shields.io/badge/arXiv-<NUMBER>.<NUMBER>-b31b1b.svg?style=flat&#41;]&#40;https://arxiv.org/abs/<NUMBER>.<NUMBER>&#41;)

[comment]: <> (```)

[comment]: <> (@inproceedings{cool,)

[comment]: <> (  title={bla},)

[comment]: <> (  author={Sam Lerman and Chenliang Xu},)

[comment]: <> (  booktitle={bla},)

[comment]: <> (  year={2022},)

[comment]: <> (  url={https://openreview.net})

[comment]: <> (})

[comment]: <> (```)

```bibtex
@article{cool,
  title   = {UnifiedML: A Unified Framework For Intelligence Training},
  author  = {Lerman, Sam and Xu, Chenliang},
  journal = {arXiv preprint arXiv:2203.08913},
  year    = {2022}
}
```

[comment]: <> (```bibtex)

[comment]: <> (@inproceedings{UML,)

[comment]: <> (  title={UnifiedML: A Unified Framework For Intelligence Training},)

[comment]: <> (  author={Lerman, Sam and Xu, Chenliang},)

[comment]: <> (  booktitle={booktitle},)

[comment]: <> (  year={2022},)

[comment]: <> (  url={https://openreview.net})

[comment]: <> (})

[comment]: <> (```)

If you use this work, please give us a star :star: and be sure to cite the above!

An acknowledgment to [Denis Yarats](https://github.com/denisyarats), whose excellent [DrQV2 repo](https://github.com/facebookresearch/drqv2) inspired much of this library and its design.

# :open_umbrella: Unified Learning?

Yes.

All agents support discrete and continuous control, classification, and generative modeling.

See example scripts of various configurations [below](#mag-sample-scripts).

# :wrench: Setting Up

Let's get to business.

## 1. Clone The Repo

```console
git clone git@github.com:agi-init/UnifiedML.git
cd UnifiedML
```

## 2. Gemme Some Dependencies

All dependencies can be installed via [Conda](https://docs.conda.io/en/latest/miniconda.html):

```console
conda env create --name ML --file=Conda.yml
```

## 3. Activate Your Conda Env.

```console
conda activate ML
```

[comment]: <> (For GPU support, you may need to install Pytorch with CUDA from https://pytorch.org/get-started/locally/.)

[comment]: <> (```console)

[comment]: <> (conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch)

[comment]: <> (```)

# :joystick: Installing The Suites

## 1. Atari Arcade

<p align="left">
<img src="https://i.imgur.com/ppm4LJw.jpg" width="320">
<br><i>A collection of retro Atari games.</i>
</p>

You can install via ```AutoROM``` if you accept the license.

```console
pip install autorom
AutoROM --accept-license
```
Then:
```console
mkdir ./Datasets/Suites/Atari_ROMS
AutoROM --install-dir ./Datasets/Suites/Atari_ROMS
ale-import-roms ./Datasets/Suites/Atari_ROMS
```

## 2. DeepMind Control

Comes pre-installed! For any issues, consult the [DMC repo](https://github.com/deepmind/dm_control).

<p align="left">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=rAai4QzcYbs" target="_blank"><i>:arrow_forward: Click to play</i></a><br>
<a href="http://www.youtube.com/watch?feature=player_embedded&v=rAai4QzcYbs" target="_blank">
<img src="https://i.imgur.com/vzNmMMQ.png" alt="Play video" width="310" />
</a>
<br><i>Video of different tasks in action.</i>
</p>

## 3. Classify

<p align="left">

[comment]: <> (<img src="https://i.imgur.com/F633xwk.png" width="320">)

[comment]: <> (<br><i>Alpaca or llama? Donkey or mule? Roses or kale? — iNaturalist</i><br><br>)
<img src="https://i.imgur.com/N1st6uO.png" width="320">
<br><i>Eight different ladybug species in the iNaturalist dataset.</i>

[comment]: <> (<br><br>)

[comment]: <> (<img src="https://i.imgur.com/etoaz2b.png" width="320">)

[comment]: <> (<br><i>Samples of images from the CIFAR-100 dataset.</i>)

[comment]: <> (<br><br><img src="https://i.imgur.com/E1v1jvm.jpg" width="320">)

[comment]: <> (<br><i>Samples of images from the Tiny-ImageNet dataset.</i>)
</p>

[comment]: <> (Comes preinstalled.)

[comment]: <> (No additional preparation needed. All datasets download automatically.)

[All datasets](Hyperparams/task/classify) come ready-to-use :white_check_mark:

That's it.

[comment]: <> (All datasets come preinstalled :white_check_mark:)

# :file_cabinet: Key files

```Run.py``` handles learning and evaluation loops, saving, distributed training, logging, plotting.

```Environment.py``` handles rollouts.

```./Agents``` contains self-contained agents.

# :mag: Sample scripts

### RL

<details>
<summary>
:mag: <i>Click to interact</i>
</summary>
<br>

Humanoid example:
```console
python Run.py task=dmc/humanoid_walk
```

[DrQV2](https://arxiv.org/abs/2107.09645) Agent in Atari:
```console
python Run.py Agent=Agents.DrQV2Agent task=atari/mspacman
```

[SPR](https://arxiv.org/abs/2007.05929) Agent in DeepMind Control:
```console
python Run.py Agent=Agents.SPRAgent task=dmc/walker_walk
```

As of now, all agents are *visual*, that is, they observe pixel-based inputs.

Save videos with ```log_video=true```.

Achieves [top scores](#bar_chart-agents--performances) in data-efficient RL from images across Atari and DMC.

More in-depth logs can be toggled with ```agent.log=true```.

</details>

### Classification 

<details>
<summary>
:mag: <i>Click to ascertain</i>
</summary>
<br>

CNN on MNIST:

```console
python Run.py task=classify/mnist 
```

Since this is *Unified*ML, there are a few noteworthy variations.

**Variations**

```RL=false``` is the default for ```classify``` tasks. Keeps training at **standard** supervised-only classification.

With ```RL=true```, an **augmented RL** update joins the supervised learning update $\text{s.t. } reward = -error$ (**experimental**).

Alternatively, and interestingly, ```supervise=false``` will *only* supervise via RL $reward = -error$. This is **pure-RL** training and actually works!

[comment]: <> (*Note:* ```RL=false``` sets training to standard supervised-only classification. Without ```RL=false```, an additional RL update joins the supervised learning update s.t. $reward = -error$.)

[comment]: <> (Alternatively, and interestingly, ```supervise=false``` will *only* supervise via RL $reward = -error$ &#40;**experimental**&#41;. This is pure-RL training and actually works.)

[comment]: <> (with a simple CNN and some small random crop transforms.)

[comment]: <> (*Note:* ```RL=false``` sets training to standard supervised-only classification.)

[comment]: <> (Without ```RL=false```, an **Augmented RL** update joins the supervised learning update s.t. $reward = -error$.)

[comment]: <> (**Pure-RL** Alternatively, and interestingly, ```supervise=false``` will *only* supervise via RL $reward = -error$ &#40;*experimental*&#41;. This is pure-RL training and actually works.)

[comment]: <> (The latent optimization could also be done over a learned parameter space as in POPLIN &#40;Wang and Ba, 2019&#41;, which lifts the domain of the optimization problem eq. &#40;1&#41; from Y to the parameter space of a fully-amortized neural network. This leverages the insight that the parameter space of over-parameterized neural networks can induce easier non-convex optimization problems than in the original space, which is also studied in Hoyer et al. &#40;2019&#41;.)

**Important features** Many popular features are unified in this library and generalized across RL/CV/generative domains, with more being added: 

* Evaluation with [exponential moving average (EMA)](https://arxiv.org/pdf/1803.05407.pdf) of params can be toggled with the ```ema=true``` flag; customize the decay rate with ```ema_decay=```. 
  
* See [Custom Architectures](#custom-architectures) for mix-and-matching custom or pre-defined (*e.g.* ViT, ResNet50) architectures via the command line syntax. 
  
* Different optimizations [can be configured](#custom-optimization) too.
  
* As well as [Custom Datasets](#custom-datasets). 
  
* Training with [weight decay](https://arxiv.org/abs/1711.05101) can be toggled via ```weight_decay=```. 
  
* A [cosine annealing learning rate schedule](https://arxiv.org/abs/1608.03983) can be applied for $N$ epochs (or episodes in RL) with ```lr_decay_epochs=```. 
  
* And [TorchVision transforms](https://pytorch.org/vision/stable/transforms.html) can be passed in as dicts via ```transform=```. 
  
For example,

```console
python Run.py task=classify/cifar10 ema=true weight_decay=0.01 transform="{RandomHorizontalFlip:{p:0.5}}" Eyes=Blocks.Architectures.ResNet18
```

The above returns a $93$% on CIFAR-10 with a ResNet18, which is pretty good. Changing datasets/architectures is as easy as modifying the corresponding parts ```task=``` and ```Eyes=``` of the above script.

And if you set ```supervise=false```, we get a $94$%... vis-à-vis pure-RL. 

[comment]: <> (Rollouts fill up data in an online fashion, piecemeal, until depletion &#40;all data is processed&#41; and gather metadata like past predictions, which may be useful for curriculum learning.)

[comment]: <> (Automatically toggles ```offline=true``` by default, but can be set to ```false``` if past predictions or "streaming" data is needed.)

This library is meant to be useful for academic research, and out of the box supports [many datasets](Hyperparams/task/classify), including 
* Tiny-ImageNet (```task=classify/tinyimagenet```), 
* iNaturalist, (```task=classify/inaturalist```),
* CIFAR-100 (```task=classify/cifar100```), 
* & [more](Hyperparams/task/classify), normalized and no manual preparation needed

</details>

### Offline RL

<details>
<summary>
:mag: <i>Click to recall</i>
</summary>
<br>

From a saved experience replay, sans additional rollouts:

```console
python Run.py task=atari/breakout offline=true
```

Assumes a replay [is saved](#saving).

Implicitly treats ```replay.load=true``` and ```replay.save=true```, and only does evaluation rollouts.

Is true by default for classification, where replays are automatically downloaded.

</details>

### Generative Modeling

<details>
<summary>
:mag: <i>Click to synthesize</i>
</summary>
<br>

Via the ```generate=true``` flag:
```console
python Run.py task=classify/mnist generate=true
```

<p align="left">
<img src="https://i.imgur.com/N1st6uO.png" width="320">
<br><i>Synthesized MNIST images, conjured up and imagined by a simple MLP.</i>
</p>

Saves to ```./Benchmarking/<experiment>/<Agent name>/<task>_<seed>_Video_Image/```.

Defaults can be easily modified with custom architectures or even datasets as elaborated in [Custom Architectures](#custom-architectures) and [Custom Datasets](#custom-dataset). Let's try the above with a CNN Discriminator:

```console
python Run.py task=classify/mnist generate=true Discriminator=CNN
```

Or a ResNet18:

```console
python Run.py task=classify/mnist generate=true Discriminator=Resnet18
```

Or let's speed up training by turning off the default image augmentation, which is overkill anyway for this simple case:

```console
python Run.py task=classify/mnist generate=true Aug=Identity
```

```Aug=Identity``` substitutes the default random cropping image-augmentation with the Identity function, thereby disabling it.

Generative mode implicitly treats training as [offline](#offline-rl), and assumes a replay [is saved](#saving) that can be loaded. 

As long as a dataset is available or a replay has been saved, ```generate=true``` will work for any defined task, making it a powerful hyper-parameter that can work regardless of domain.

[comment]: <> (TODO: set defualts for generate in Run.py/Environment.py automatically)
Can even work with RL tasks (due to frame stack, the generated images are technically multi-frame videos), but make sure to change some of the default settings to speed up training, as per below:

```console
python Run.py task=atari/breakout generate=true evaluate_episodes=1 action_repeat=1
```

Make sure you have [saved a replay](#saving) before doing this.

</details>

[comment]: <> (ensemble could help this:)

[comment]: <> (Extensions. Analyzing and extending the amortization components has been a key development in AVI methods. Cremer et al. &#40;2018&#41; investigate suboptimality in these models are categorize it as coming from an amortization gap where the amortized model for eq. &#40;30&#41; does not properly solve it, or the approximation gap where the variational posterior is incapable of approximating the true distribution. Semi-amortization plays a crucial role in addressing the amortization gap and is explored in the semi-amortized VAE &#40;SAVAE&#41; by)

[comment]: <> (Kim et al. &#40;2018&#41; and iterative VAE &#40;IVAE&#41; by Marino et al. &#40;2018&#41;.)

### Saving
<details>
<summary>
:mag: <i>Click to load</i>
</summary>
<br>

**Agents** can be saved periodically or loaded with the ```save_per_steps=``` or ```load=true``` flags, and are automatically saved at end of training with ```save=true``` by default.

```console
python Run.py save_per_steps=100000 load=true
```

An **experience replay** can be saved or loaded with the ```replay.save=true``` or ```replay.load=true``` flags.

```console
python Run.py replay.save=true replay.load=true
```

Agents and replays save to ```./Checkpoints``` and ```./Datasets/ReplayBuffer``` respectively per *a unique experiment*, otherwise overriding.

*A unique experiment* is distinguished by the flags: ```experiment=```, ```Agent=```, ```task=```, and ```seed=```.

Replays also save uniquely w.r.t. a date-time. In case of multiple saved replays per a unique experiment, the most recent is loaded.

Careful, without ```replay.save=true``` a replay, whether new or loaded, will be deleted upon terminate, except for the default offline classification replays.

</details>

### Distributed

<details>
<summary>
:mag: <i>Click to disperse</i>
</summary>
<br>

The simplest way to do distributed training is to use the ```parallel=true``` flag,

```console
python Run.py parallel=true 
```

which automatically parallelizes the Encoder's "Eyes" across all visible GPUs. The Encoder is usually the most compute-intensive architectural portion.

To share whole agents across multiple parallel instances and/or machines,

<details>

<summary><i>Click to expand :open_book: </i></summary>

<br>

you can use the ```load_per_steps=``` flag.

For example, a data-collector agent and an update agent,

```console

python Run.py learn_per_steps=0 replay.save=true load_per_steps=1

```

```console

python Run.py offline=true replay.offline=false replay.save=true replay.load=true save_per_steps=2

```

in concurrent processes.

Since both use the same experiment name, they will save and load from the same agent and replay, thereby emulating distributed training. Just make sure the replay from the first script is created before launching the second script. **Highly experimental!**

Here is another example of distributed training, via shared replays:

```console
python Run.py replay.save=true 
```

Then, in a separate process, after that replay has been created:

```console
python Run.py replay.load=true replay.save=true 
```

[comment]: <> (It's a bit finicky; there are a few timing delicacies that I don't account for. I recommend to wait until at least 1 episode for the first script's replay to be created before launching the second script. This is not meant as a deployable means of distributed training. It just happens to work, incidentally, sort of.)

</details>

</details>

### Recipes

<details>
<summary>
:mag: <i>Learn to cook</i>
</summary>
<br>

</details>

### Custom Architectures

<details>
<summary>
:mag: <i>Click to construct</i>
</summary>
<br>

We use a rich and expressive command line syntax for selecting and customizing architectures such as those defined in ```./Blocks/Architectures```.

ResNet18 on CIFAR-10:

```console
python Run.py task=classify/cifar10 Eyes=Blocks.Architectures.ResNet18 
```

[comment]: <> (TODO: MiniViT, ViT)
Atari with ViT:

```console
python Run.py Eyes=Blocks.Architectures.ViT +recipes.encoder.eyes.patch_size=7
```

[comment]: <> (TODO: Eyes, Ears, etc. recipes -> hands)
Shorthands like ```Eyes``` and ```pool``` make it easy to plug and play custom architectures, but all of an agent's architectural parts can be accessed, mixed, and matched with the ```recipes.``` keyword or their [corresponding shorthands](Hyperparams/args.yaml#L166).

Generally, the rule of thumb is capital names for paths to classes (such as ```Eyes=Blocks.Architectures.MLP```) and lowercase names for shortcuts to tinker with model args (such as ```+eyes.depth=1```).


<details>
<summary><i>See more examples :open_book: </i></summary>
<br>

CIFAR-10 with ViT:

```console
python Run.py Eyes=ViT task=classify/cifar10 ema=true weight_decay=0.01 +eyes.depth=6 +eyes.out_channels=512 +eyes.hidden_dim=512 transform="{RandomCrop:{size:32,padding:4},RandomHorizontalFlip:{}}"
```

[comment]: <> (TODO: Generator/Discriminator shorthands, with default input_shape=${obs_shape})

Here is a more complex example, disabling the Encoder's flattening of the feature map, and instead giving the Actor and Critic unique Attention Pooling operations on their trunks to pool the unflattened features. The ```Null``` architecture disables that flattening component, though in this case it's not actually necessary since the ```AttentionPool``` architecture has adaptive input broadcasting - I'm pointing it out because in the general case, it might be useful.

```console
python Run.py task=classify/mnist Q_trunk=AttentionPool Pi_trunk=AttentionPool Pool=Blocks.Architectures.Null
```

It is recommended to use the full path for ```Blocks.Architectures.Null``` or to put it in quotes ```'Pool="Null"'``` or else Hydra may confuse it with the default ```null <-> None``` grammar. 

Here is a nice example of the critic using a small CNN for downsampling features:

```console
python Run.py task=classify/mnist Q_trunk=CNN +q_trunk.depth=1
```

A CNN Actor and Critic:
```console
python Run.py Q_trunk=CNN Pi_trunk=CNN +q_trunk.depth=1 +pi_trunk.depth=1 'Pool="Null"'
```

[comment]: <> (<details>)

[comment]: <> (<summary><i>See even more examples :open_book: </i></summary>)

[comment]: <> (<br>)

[comment]: <> (Here's how you can load another saved agent's encoder from a pre-configured agent checkpoint ```<path>```:)

[comment]: <> (```)

[comment]: <> (python Run.py Eyes=Utils.load +recipes.encoder.eyes.path=<path> +recipes.encoder.eyes.attr=encoder.Eyes)

[comment]: <> (```)

[comment]: <> (You can imagine training a GAN CNN and then seamlessly using it for RL.)

[comment]: <> (<br>)

[comment]: <> (</details>)

*A little secret*, but pytorch code can be passed directly too via quotes:

```console
python Run.py "recipes.encoder.eyes='CNN(kwargs.input_shape,32,depth=3)'"
```
```console
python Run.py "eyes='torch.nn.Conv2d(kwargs.input_shape[0],32,kernel_size=3)'"
```

Some blocks have default args which can be accessed with the ```kwargs.``` interpolation shown above.

An intricate example of the expressiveness of this syntax:
```console
python Run.py Optim=torch.optim.SGD 'Pi_trunk="nn.Sequential(MLP(input_shape=kwargs.input_shape, output_dim=kwargs.output_dim),nn.ReLU(inplace=True))"' lr=0.01
```

Both the uppercase and lowercase syntax support direct function calls in lieu of their usual syntax, with function calls distinguished by the quotes and parentheticals.

The parser automatically registers the imports/class paths in ```Utils``` in both the uppercase and lowercase syntax, including modules/classes ```torch```, ```torch.nn```, and architectures/paths in ```./Blocks/Architectures/``` like ```CNN```.

</details>

Of course, it's always possible to just modify the library code itself, which may be easier. See for example the two Encoder variants in ```./Blocks/Encoders.py```.

To make your own architecture mix-and-matchable, just put it in a pytorch module with initialization options for ```input_shape``` and ```output_dim```, as in the architectures in ```./Blocks/Architectures```.

[comment]: <> (TODO repr_shape)

</details>

### Custom Optimizers

<details>
<summary>
:mag: <i>Click to search/explore</i>
</summary>
<br>

Optimization parts can be accessed *e.g.* 

```console
python Run.py Optim=Utils.torch.optim.SGD lr=0.1
```

or via the expressive recipe interface described in [Custom Architectures](#custom-architectures):

```console
python Run.py optim=torch.optim.SGD lr=0.1
```

```console
python Run.py "optim='torch.optim.SGD(kwargs.params, lr=0.1)'"
```

</details>

### Custom Dataset

<details>
<summary>
:mag: <i>Click to read/parse</i>
</summary>
<br>

</details>

### Experiment naming, plotting

<details>
<summary>
:mag: <i>Click to see</i>
</summary>
<br>

Plots automatically save to ```./Benchmarking/<experiment>/```, the default experiment is ```experiment=Exp```.

```console
python Run.py
```

:chart_with_upwards_trend: :bar_chart: in ```./Benchmarking/Exp/```

Optionally plot multiple experiments in a unified figure with ```plotting.plot_experiments=```.

```console
python Run.py experiment=Exp2 plotting.plot_experiments="['Exp', 'Exp2']"
```

Alternatively, you can call ```Plot.py``` directly

```console
python Plot.py plot_experiments="['Exp', 'Exp2']"
```

to generate plots. Here, the ```<experiment>``` directory name will be the underscore_concatenated union of all experiment names ("```Exp_Exp2```").

Plotting also accepts regex expressions. For example, to plot all experiments with ```Exp``` in the name:

```console
python Plot.py plot_experiments="['.*Exp.*']"
```

Another option is to use [WandB](https://wandb.ai/), which is supported by UnifiedML:

```console
python Run.py logger.wandb=true
```

You can connect UnifiedML to your WandB account by first running ```wandb login``` in your Conda environment.

To do a hyperparameter sweep, just use the ```-m``` flag.
```console
python Run.py -m task=atari/pong,classify/mnist seed=1,2,3 
```

</details>

[comment]: <> (The above will sweep over random seeds 1, 2, and 3, and whether to use EMA.)

# :bar_chart: Agents & Performances

# :interrobang: How is this possible

We use our new Creator framework to unify RL discrete and continuous action spaces, as elaborated in our [paper](https://arxiv.com).

Then we frame actions as "predictions" in supervised learning. We can even augment supervised learning with an RL phase, treating reward as negative error.

For generative modeling, well, it turns out that the difference between a Generator-Discriminator and Actor-Critic is rather nominal.

[comment]: <> (![alt text]&#40;https://i.imgur.com/Yf8ltyI.png&#41;)

<img width="80%" alt="flowchart" src="https://i.imgur.com/nMUR9Ue.png">

[comment]: <> (</p>)

[comment]: <> (<img width="80%" alt="flowchart" src="https://i.imgur.com/RM52cfJ.png?1">)

# :mortar_board: Pedagogy and Research

All files are designed for pedagogical clarity and extendability for research, to be useful for educational and innovational purposes, with simplicity at heart.

# :people_holding_hands: Contributing

Please support financially: <br>
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg?style=flat-square)](https://www.paypal.com/cgi-bin/) <br>

We are a nonprofit, single-PhD student team. If possible, compute resources appreciated.

[comment]: <> (Our work will go towards helping nature through AI, making academic-level research accessible to all, and simplifying, bridging, and unifying the vast array of problem domains in our field.)

Feel free to [contact **agi.\_\_init\_\_**](mailto:agi.init@gmail.com).

I am always looking for collaborators. Don't hesitate to volunteer in any way to help realize the full potential of this library.

# Note

[comment]: <> (*While UnifiedML V.0 is a fully-realized, self-complete library, we note that it is also a ground for expansion beyond what is presented, with more performative breadth and depth on the way.*)

### If you are only interested in the RL portion,

Check out our [**UnifiedRL**](https:github.com/agi-init/UnifiedRL) library.

It does with RL to this library what PyCharm does with Python to IntelliJ, i.e., waters it down mildly and rebrands a little.~

<hr class="solid">

[MIT license Included.](MIT_LICENSE)



  