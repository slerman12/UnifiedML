from Sweeps.Templates import template


runs = template('XRD')


# Generalization To LATT
runs.XRD.sweep = [
    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='050_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/050/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='060_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/060/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='070_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/070/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='080_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/080/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='090_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/090/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='095_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/095/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='098_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/098/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='102_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/102/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='105_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/105/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='110_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/110/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='120_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/120/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='130_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/130/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='140_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/140/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='150_Large-Soup_${test_dataset.num_classes}-Way'
    experiment='Large-Soup_No-Pool-CNN'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/150/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180"""
]

# Original Summary
# runs.XRD.plots = [
#     ['CNN_optim.*', 'MLP_optim_.*'],
# ]
# runs.XRD.tasks = [
#     'Soup-50-50.*', 'Synthetic.*'
# ]

# For Suites Aggregation
# runs.XRD.plots = [
#     ['CNN', 'MLP', 'No-Pool-CNN'],
# ]
# runs.XRD.tasks = [
#     'Large.*', 'Mix-Soup.*'
# ]

# runs.XRD.plots = [
#     ['CNN', 'MLP', 'No-Pool-CNN'],
# ]
# runs.XRD.tasks = [
#     'PS1_.*', 'Large_.*', 'Mix_.*', 'Large-Soup_.*', 'Mix-Soup_.*'
# ]
# runs.XRD.tasks = [
#     'Large-Soup_.*'
# ]

# runs.XRD.title = 'RRUFF'

runs.XRD.plots = [
    # ['.*icsd.*'],
    # ['.*_icsd.*'],
    # ['.*_nonicsd.*'],
    # ['.*MP'],
    ['Large-Soup_No-Pool-CNN']
]

runs.XRD.title = 'LATT'
runs.XRD.sftp = True
