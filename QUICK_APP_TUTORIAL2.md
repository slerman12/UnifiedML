# Use it in an app

### Install

```console
pip install UnifiedML
```

### Command-line example

**App.py:**

```python
from torch import nn

model = nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))
```

**Run it:**

```console
ML Predictor=App.model Dataset=CIFAR10
```

### Pure-Code example

**App.py:**
```python
from torch import nn

import UnifiedML

model = nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))

UnifiedML.launch(Predictor=model, Dataset='CIFAR10')
```

**Run it:**

```console
python App.py
```

### Adaptive shaping

UnifiedML automatically detects the signature of your model class and passes in the shapes accordingly.

**App.py:**

```python
from torch import nn

import UnifiedML

# Simple Pytorch class with unknown shapes
class Predictor(nn.Module): 
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.model = nn.Sequential(nn.Linear(in_features, 128), nn.Linear(128, out_features))

    def forward(self, x):
        return self.model(x)

UnifiedML.launch(Predictor=Predictor, Dataset='CIFAR10')
```

```console
python App.py
```

Inferrable signature arguments: ```in_shape```, ```out_shape```, ```in_features```, ```out_features```, ```in_channels```, ```out_channels```, ```in_dim```, and ```out_dim```.

### Acceleration

With the ```accelerate=True``` flag you can leverage UnifiedML's training accelerations, which include hard-disk memory mapping with adaptive truly-shared RAM allocation, automatic 16-bit precision, and multi-GPU parallel batch training. For RL, there's no downside. For image classification, extra hard disk memory is needed to store the re-formatted dataset. 

```console
python App.py accelerate=true
```

**or in App.py:**

```python
...

UnifiedML.launch(accelerate=True)
```

### Recipes example - Training CIFAR-10 with ResNet18

Define your own recipe in a ```.yaml``` file like this one:

Then use ```task=``` to select the recipe:

```console
ML task=cifar_recipe accelerate=true
```

This recipe exactly trains CIFAR-10 to 94% accuracy in 5 minutes on 1 GPU. 

### Hyperparams

All hyperparams can be passed in via command-line, code, or recipe.

**Here's how to write the same program in 3 different ways:**

#### 1. Command-line

```console
ML task=classify Dataset=MNIST Eyes=CNN +eyes.depth=5
```

#### 2. Code

**App.py:**
```python
import UnifiedML
UnifiedML.launch('+eyes.depth=5', task='classify', Dataset='MNIST', Eyes='CNN')
```
**Run it:**

```console
python App.py
```

#### 3. Recipe

**Recipe.yaml:**

**Run it:**

```console
ML task=Recipe
```

---

**Any combination thereof is also valid.** 

The order of priority is command-line > code > recipe. 

**App.py:**
```python
import UnifiedML
UnifiedML.launch(task='recipe', Dataset='MNIST')
```
**Run it:**

```console
python App.py Eyes=CNN +eyes.depth=5
```

The ```+hyperparam.``` syntax is used to modify arguments of flag ```Hyperparam```. We reserve ```Uppercase=Path.To.Class``` for the class itself and ```+lowercase.key=value``` for argument tinkering, as in the example above.

### Plotting

Let's consider our [CIFAR-10](#recipes-example---training-cifar-10-with-resnet18) example from earlier.

```console
ML task=cifar_recipe accelerate=true
```

We can plot the results of this as follows:

```console
Plot task=cifar_recipe
```

The following plots save in ```Benchmarking/```:

We can use flags like ```experiment=``` to distinguish experiments. The recipe ```cifar_recipe``` includes such a flag.

### Recipes - Train a humanoid to walk from images, 1.2x faster than the SOTA DrQV2

Define your own recipe in a ```.yaml``` file like this one:

**App.py:**

```python
import UnifiedML
UnifiedML.launch(task='humanoid_from_images')
```

```console
python App.py
```

**Generate plots:**

SOTA scores at 1.2x the speed.

### Recipes - Atari to human-normalized score in 1m steps

**Train:**

```console
ML -m task=RL +env.game=...,pong,...
```

The ```-m``` flag sweeps over comma-separated hyperparams. For more sophisticated sweeps, check out [SweepsAndPlots]().

**Plot:**

```console
Plot task=RL +env.game=...,pong,...
```

```task=RL``` refers to one of the pre-defined recipes/tasks in [UnifiedML/Hyperparams/task](). These can be accessed from outside apps, as well as any other UnifiedML search paths and features.

### Recipes - DCGAN in 5 minutes

**Train:**

```console
ML task=dcgan
```

[//]: # (Plots, reel)
[//]: # (caption: something .. as saved in ```Benchmarking/```.)