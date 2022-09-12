from Sweeps.Templates import template


runs = template('XRD')

runs.XRD.sweep = [
    'experiment="Test2_K80" task=classify/mnist train_steps=2000 gpu=K80',
    'experiment="Test2_V100" task=classify/mnist train_steps=2000 gpu=V100',
    'experiment="Test2_A100" task=classify/mnist train_steps=2000 gpu=A100',
    'experiment="Test2_RTX_Lab" task=classify/mnist train_steps=2000 gpu=RTX lab=true',
]

runs.XRD.plots = [
    ['Test2_.*'],
]
runs.XRD.title = 'A Good Ol\' Test: CUDA 10.2 & 11.3'
