from Sweeps.Templates import template


runs = template('XRD')

runs.XRD.sweep = [
    # # Large-Soup, No-Pool-CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.NoPoolCNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Large-Soup_${dataset.num_classes}-Way'
    # experiment='No-Pool-CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0.5]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # mem=100""",
    #
    # Large-Soup, CNN
    """task=classify/custom
    Dataset=XRD.XRD
    Aug=Identity
    Trunk=Identity
    Eyes=XRD.CNN
    Predictor=XRD.Predictor
    batch_size=256
    standardize=false
    norm=true
    task_name='Large-Soup_${dataset.num_classes}-Way'
    experiment='CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    +dataset.num_classes=7,230
    train_steps=5e5
    save=true
    logger.wandb=true
    lab=true
    num_workers=1
    mem=30""",

    # Large-Soup, MLP
    """task=classify/custom
    Dataset=XRD.XRD
    Aug=Identity
    Trunk=Identity
    Eyes=Identity
    Predictor=XRD.MLP
    batch_size=256
    standardize=false
    norm=true
    task_name='Large-Soup_${dataset.num_classes}-Way'
    experiment='MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    +dataset.num_classes=7,230
    train_steps=5e5
    save=true
    logger.wandb=true
    lab=true
    num_workers=1
    mem=30""",

    # Large, CNN
    """task=classify/custom
    Dataset=XRD.XRD
    Aug=Identity
    Trunk=Identity
    Eyes=XRD.CNN
    Predictor=XRD.Predictor
    batch_size=256
    standardize=false
    norm=true
    task_name='Large_${dataset.num_classes}-Way'
    experiment='CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0]'
    +dataset.num_classes=7,230
    train_steps=5e5
    save=true
    logger.wandb=true
    lab=true
    num_workers=1
    mem=30""",

    # Large, MLP
    """task=classify/custom
    Dataset=XRD.XRD
    Aug=Identity
    Trunk=Identity
    Eyes=Identity
    Predictor=XRD.MLP
    batch_size=256
    standardize=false
    norm=true
    task_name='Large_${dataset.num_classes}-Way'
    experiment='MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0]'
    +dataset.num_classes=7,230
    train_steps=5e5
    save=true
    logger.wandb=true
    lab=true
    num_workers=1
    mem=30""",

    # # PS1, CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.CNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='PS1_${dataset.num_classes}-Way'
    # experiment='CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_ps1/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # mem=20""",
    #
    # # PS1, MLP
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=Identity
    # Predictor=XRD.MLP
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='PS1_${dataset.num_classes}-Way'
    # experiment='MLP'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_ps1/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # mem=20""",
]

# Create universal replays
# runs.XRD.sweep = [
#     # Large-Soup, MLP
#     """task=classify/custom
#     Dataset=XRD.XRD
#     Aug=Identity
#     Trunk=Identity
#     Eyes=Identity
#     Predictor=XRD.MLP
#     batch_size=256
#     standardize=false
#     norm=true
#     task_name='Large-Soup_${dataset.num_classes}-Way'
#     experiment='MLP'
#     '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
#     +'dataset.train_eval_splits=[1, 0.5]'
#     +dataset.num_classes=7,230
#     train_steps=1
#     save=false
#     num_workers=1
#     logger.wandb=true
#     lab=true""",

    # # Large, MLP
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=Identity
    # Predictor=XRD.MLP
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Large_${dataset.num_classes}-Way'
    # experiment='MLP'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=1
    # save=false
    # num_workers=1
    # logger.wandb=true
    # lab=true""",
    #
    # # PS1, CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.CNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='PS1_${dataset.num_classes}-Way'
    # experiment='CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_ps1/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=1
    # save=false
    # num_workers=1
    # logger.wandb=true
    # lab=true""",
# ]

runs.XRD.plots = [
    ['CNN', 'MLP', 'No-Pool-CNN'],
]
runs.XRD.tasks = [
    'PS1.*', 'Large.*'
]
runs.XRD.title = 'RRUFF'
