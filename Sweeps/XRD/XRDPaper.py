from Sweeps.Templates import template


runs = template('XRD')

runs.XRD.sweep = [
    'experiment=Test1 task=classify/mnist train_steps=2000'
]

runs.XRD.plots = [
    ['Test1'],
]
runs.XRD.title = 'A Good Ol\' Test'
