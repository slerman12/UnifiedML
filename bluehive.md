### Quick Links

- [Setup](#setting-up-unifiedml-on-bluehive)

- [Structure](#structure)

- [Launching Runs](#rocket-launching)

- [Specifying Runs](#specifying-runs)

# Setting Up UnifiedML On Bluehive

Connect to the University VPN.

I use:

```console
python vpn.py
```

But the typical way is to manually connect via the [Cisco AnyConnect app recommended by the University](https://tech.rochester.edu/services/remote-access-vpn/).

Then login to Bluehive:

```console
ssh <username>@bluehive.circ.rochester.edu
```

Start a persistent session using [tmux](https://en.wikipedia.org/wiki/Tmux), a great tool that persists your work session even when you exit/disconnect from Bluehive.

```console
# Enables use of tmux
module load tmux

# Creates a new session called ML
tmux new -s ML
```

Later, you can re-open that session with:

```console
# Opens an existing session called ML
tmux attach -t ML
```

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) into your ```/home/<username>``` directory:

```console
cd /scratch/<username>

# Downloads the installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./Miniconda.sh

# Installs Miniconda 
# Select /home/<username> and agree to init conda when prompted.
sh Miniconda.sh
```

For changes to take effect, you have to start a new tmux session. ```Control-D``` out of the current tmux session and create a new one as above:

```console
tmux new -s ML
```

Now go to your ```/scratch/<username>``` directory:

```console
cd /scratch/<username>
```

Install UnifiedML [following the instructions here](https://www.github.com/agi-init/UnifiedML#wrench-setting-up).

When choosing a CUDA version, I've found ```11.3``` to work best across the different Bluehive GPU types (K80, RTX, V100, and A100). The K80 requires a lower CUDA, I use ```10.2``` (the one installed by default for UnifiedML) and a separate Conda environment for that one. The launch script can select the Conda environment adaptively depending on which GPU gets assigned after launching (more details below).

```console
# CUDA 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

On Bluehive, you must enable ```gcc``` for torchvision to work:

```console
module load gcc
```

The way to use Bluehive, is to queue up jobs on specific GPU nodes with an "```sbatch script```". I use an automated pipeline that generates and calls sbatch scripts according to my specified runs and hyper-parameters and selected GPUs, etc. It automatically connects to VPN and Bluehive and launches my desired jobs.

# Structure

I specify my runs in: [```sweeps_and_plots.py```](sweeps_and_plots.py).
- I also specify how to plot them / their corresponding plots.

I launch them with ```python launch_bluehive.py```
- Which connects to Bluehive and then calls [```sbatch.py```](sbatch.py) on Bluehive to deploy jobs.

When all is said and done, I plot locally from Bluehive by running: ```python plot_bluehive_and_lab.py```
- Connects to servers and Bluehive, downloads the benchmarking data specified in [```sweeps_and_plots.py```](sweeps_and_plots.py), and plots accordingly.

- UnifiedML also supports plotting to [WandB](https://wandb.ai/)'s online dashboards in real-time if you want. See [Experiment naming, plotting](https://github.com/AGI-init/UnifiedML#experiment-naming-plotting).

Below, I'll go over how to use [```launch_bluehive.py```](launch_bluehive.py).

# :rocket: Launching

Set the [username](launch_bluehive.py#L14) in [```launch_bluehive.py```](launch_bluehive.py). For example,

```ruby
username = 'slerman'
```

Update the [conda path and name](launch_bluehive.py#L29) in [```launch_bluehive.py```](launch_bluehive.py). For example,

```ruby
conda = 'source /home/{username}/miniconda3/bin/activate ML'
```

Likewise, update the [username, conda path, and conda name](Hyperparams/sbatch.yaml#L6-L7) in [```Hyperparams/sbatch.yaml```](Hyperparams/sbatch.yaml). For example,

```ruby
username: 'slerman'
conda: 'source /home/slerman/miniconda3/bin/activate ML'
```

In [```sbatch.py```](sbatch.py), set your your [conda path and name](sbatch.py#L63) and [WandB API key](sbatch.py#L65). For example,

```ruby
cuda = f'source /home/{args.username}/miniconda3/bin/activate ML'

...

wandb_login_key = '10111111yep11111111111111111111111111110'
```

* Optional: Alternatively, specify multiple Conda envs depending on CUDA version by defining ```conda```/```cuda``` with [lines 54 and 57](sbatch.py#L54-L57) and pairing the GPU name with the corresponding Conda env.

* From WandB's docs: "**Where do I find my API key?** Once you've signed in to www.wandb.ai, the API key will be on the [Authorize page](https://wandb.ai/authorize)." Or you can comment out [line 77](sbatch.py#L77) if you aren't using a WandB account.

Set the [username](plot_bluehive_&_lab.py#L42) in [```plot_bluehive_&_lab.py```](plot_bluehive_&_lab.py). For example,

```ruby
username = 'slerman'
```

Add the [```sbatch.py```](sbatch.py) file to the root directory of your ```UnifiedML``` directory on Bluehive (```/scratch/<username>/UnifiedML```). You can do this by SFTP'ing for example or via git.

Add the [```Hyperparams/sbatch.yaml```](Hyperparams/sbatch.yaml) file to the ```./Hyperparams``` directory of your ```UnifiedML``` directory on Bluehive (```/scratch/<username>/UnifiedML/Hyperparams```). You can do this by SFTP'ing for example or via git.

Keep the other files in the root directory of your local ```UnifiedML``` directory.

Use the template in [```sweeps_and_plots.py```](sweeps_and_plots.py) to define some runs and launch them with:

```console
python launch_bluehive.py
```

This will launch them on Bluehive. The script should take care of connecting to VPN, then Bluehive, then queueing jobs. You will still have to approve the connection via DUO push notification on your phone.

Runs can be defined with flags corresponding to the hyperparams in [```sbatch.yaml```](Hyperparams/sbatch.yaml) in addition to the usual UnifiedML flags in ```args.yaml```. Those additional Bluehive-specific args include ```gpu```, ```mem```, and ```time``` for example. They can also accept a ```reservation_id``` if you have one, or enable a specific set of ```lab``` machines. All such specifications can be specified in the runs list of [```sweeps_and_plots.py```](sweeps_and_plots.py).

These can later be plotted with:

```console
python plot_bluehive_and_lab.py
```

Which also connects to VPN, Bluehive automatically, then downloads and plots results locally.

# Specifying Runs

An example set of runs and plots in [```sweeps_and_plots.py```](sweeps_and_plots.py):

```ruby
runs = {'Example': {
    'sweep': [
        # Sweep commands go here
        'experiment=example1 seed=1 task=classify/mnist',

        'experiment=example2 seed=2 task=classify/mnist gpu=A100'
    ],
    'plots': [
        # Sets of plots
        ['example.*'],  # Regex to plot all experiments starting with "example"
    ],

    # Plotting-related commands go here
    'sftp': True,
    'bluehive': True,
    'lab': False,
    'steps': 5e5,
    'title': 'Example',
    'x_axis': 'Step',
    'bluehive_only': [],
    'tasks': [],
    'agents': [],
    'suites': []},
}
```
---


:exclamation: &#9432;

Then to launch this, you can run:

```console
python launch_bluehive.py
```

And to plot:

```console
python plot_bluehive_and_lab.py
```

That's all.
