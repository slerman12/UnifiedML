"""
Note: not actually final
"""


from Sweeps.Templates import template


runs = template('XRD')

runs.XRD.plots = [
    ['MLP_optim_ADAM_batch_size_256.*'],
    ['.*CNN_optim_ADAM_batch_size_256.*'],
    ['ViT_optim_ADAM_batch_size_256'],
    ['ResNet18_optim_ADAM_batch_size_256'],
]
runs.XRD.tasks = ['.*230-Way.*']
runs.XRD.sftp = False
runs.XRD.bluehive = False
runs.XRD.title = 'RRUFF'
