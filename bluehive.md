## Setting Up UnifiedML On Bluehive

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

When choosing a CUDA version, I've found ```11.2``` to work best across the different Bluehive GPU types (K80, RTX, V100, and A100).

```console
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu112
```

On Bluehive, you must enable ```gcc``` for torchvision to work:

```console
module load gcc
```

The way to use Bluehive, is to queue up jobs on specific GPU nodes with an "```sbatch script```". I use an automated pipeline that generates and calls sbatch scripts according to my specified runs and hyper-parameters and selected GPUs, etc. It automatically connects to VPN and Bluehive and launches my desired jobs.

## Structure

I specify my runs in: ```sweeps_and_plots.py```.
- I also specify how to plot them / their corresponding plots.

I launch them with ```launch_bluehive.py```
- Which connects to Bluehive and then calls ```sbatch.py``` on Bluehive to deploy jobs.

When all is said and done, I plot locally from Bluehive by running: ```plot_bluehive_and_lab.py```
- Connects to servers and Bluehive, downloads the benchmarking data specified in ```sweeps_and_plots.py```, and plots accordingly.

- UnifiedML also supports plotting to WandB's online dashboards in real-time if you want. See [Experiment naming, plotting](https://github.com/AGI-init/UnifiedML#experiment-naming-plotting).

Below, I'll go over how to use ```launch_bluehive.py```.

## :rocket: Launching