# Use it in an app

### Install

```console
pip install UnifiedML
```

### Purely command-line example

**Run.py:**

```python
from torch import nn

model = nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))
```

**Run it:**

```console
ML Model=model Dataset=CIFAR10
```

### Pure-Code example

**Run.py:**

```python
from torch import nn

import ML

model = nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))

ML.launch(Model=model, Dataset='CIFAR10')
```

**Run it:**

```console
python Run.py
```

### Adaptive shaping

UnifiedML automatically detects the shape signature of your model.

**Run.py:**

```diff
from torch import nn

import ML

class Model(nn.Module): 
+   def __init__(self, in_features, out_features):
        super().__init__()
        
        self.model = nn.Sequential(nn.Linear(in_features, 128), nn.Linear(128, out_features))

    def forward(self, x):
        return self.model(x)

ML.launch(Model=Model, Dataset='CIFAR10')
```

Inferrable signature arguments include ```in_shape```, ```out_shape```, ```in_features```, ```out_features```, ```in_channels```, ```out_channels```, ```in_dim```, ```out_dim```.

```console
python Run.py
```

### Acceleration

With ```accelerate=True```, you get:
* Memory mapping
* Adaptive RAM caching
* Truly-shared RAM parallelism
* Automatic 16-bit mixed precision
* Multi-GPU automatic detection and training

```console
python Run.py accelerate=true
```

**or in Run.py:**

```python
...

ML.launch(accelerate=True)
```

For image classification, extra hard disk memory is used to store the re-formatted dataset. For RL, there's no downside.

### Image Classification Recipe - Training a ResNet18 on CIFAR10

Define recipes in a ```.yaml``` file like this one:

Then use ```task=``` to select the recipe:

```console
ML task=cifar_recipe accelerate=true
```

This recipe exactly trains CIFAR-10 to $94\%$ accuracy in 5 minutes on 1 GPU. 

* ```ResNet18``` points to this architecture [here]().

> Note: we use the terms "```task```" and "```recipe```" interchangably.

### Hyperparams

**Hyperparams can be passed in via command-line, code, recipe, or any combination thereof. Here's how to write the same program 5 different ways:** 

#### 1. Purely command-line

```console
ML task=classify Dataset=MNIST Eyes=CNN +eyes.depth=5
```

#### 2. Command line

**Run.py:**

```python
import ML
ML.launch()
```

**Run it:**

```console
python Run.py task=classify Dataset=MNIST Eyes=CNN +eyes.depth=5
```

#### 3. Code

**Run.py:**
```python
import ML
ML.launch('+eyes.depth=5', task='classify', Dataset='MNIST', Eyes='CNN')
```

```console
python Run.py
```

#### 4. Recipe

**Recipe.yaml:**

```console
ML task=Recipe
```

---

**Any combination thereof is also valid.** 

The order of hyperparam priority is command-line > code > recipe. 

#### 5. Mix

A mix of direct code, command-line, and recipe:

**Run.py:**
```python
import ML
from torchvision.datasets import MNIST

ML.launch(task='recipe', Dataset=MNIST)
```

```console
python Run.py Eyes=CNN +eyes.depth=5
```

The ```+hyperparam.``` syntax is used to modify arguments of flag ```Hyperparam```. We reserve ```Uppercase=Path.To.Class``` for the class itself and ```+lowercase.key=value``` for argument tinkering, as in the example above.

### Plotting

Let's consider our [CIFAR-10 example from earlier](#recipes-example---training-cifar-10-with-resnet18):

```console
ML task=cifar_recipe accelerate=true
```

We can plot the results as follows:

```console
Plot task=cifar_recipe
```

Corresponding plots save in ```Benchmarking/```:

We can use flags like ```experiment=``` to distinguish experiments. 

* Another option is to use [WandB]():

    ```console
    ML task=cifar_recipe accelerate=true logger.wandb=true
    ```

### RL Recipe - Train a humanoid to walk from images, 1.2x faster than the SOTA DrQV2

Define your own recipe in a ```.yaml``` file like this one:

**humanoid_from_images.yaml:**

**Run.py:**

```python
import ML
ML.launch(task='humanoid_from_images')
```

```console
python Run.py
```

**Generate plots:**

SOTA scores at 1.2x the speed.

* DrQV2Agent points [here]().

**Render a video:**

### RL Recipe - Atari to human-normalized score in 1m steps

**Train:**

```console
ML -m experiment=ALE task=RL Env=Atari +env.game=...,pong,...
```

The ```-m``` flag enables sweeping over comma-separated hyperparams, in this case a standard benchmark 26 games in the Atari ALE. For more sophisticated sweep tools, check out [SweepsAndPlots]().

**Plot:**

```console
Plot experiment=ALE
```

We can also plot it side by side with the DeepMind Control Suite RL benchmark:

**Train some tasks in the suite*:*

```console
ML -m experiment=DMC task=RL Env=Atari +env.game=...,walker_walk,...
```

**Plot:*

```console
Plot experiments=[ALE,DMC]
```

* DQNAgent points [here]().

### Generative Recipe - DCGAN in 5 minutes

**Train:**

```console
ML task=dcgan
```

[//]: # (Plots, reel)
[//]: # (caption: something .. as saved in ```Benchmarking/```.)

```task=dcgan``` refers to one of the pre-defined task recipes in [UnifiedML/Hyperparams/task](). These — like all UnifiedML recipes, search paths, and features — can be accessed from outside apps.

* DCGAN points [here]().

### Useful flags

* ```norm=true```: enables normalization 

### When to use Eyes? When to use Predictor?

### Saving/loading

### Example publication

### Example generalist agent