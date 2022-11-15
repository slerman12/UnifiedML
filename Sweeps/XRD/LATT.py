from Sweeps.Templates import template


runs = template('XRD')


# Generalization To LATT
runs.XRD.sweep = [
    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='050_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='060_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='070_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='080_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='090_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='095_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='098_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='102_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='105_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='110_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='120_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='130_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='140_No-Pool-CNN'
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
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='150_No-Pool-CNN'
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


# Generalization To LATT
runs.XRD.sweep = [
    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='050_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/050/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='060_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/060/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='070_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/070/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='080_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/080/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='090_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/090/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='095_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/095/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='098_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/098/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='102_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/102/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='105_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/105/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='110_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/110/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='120_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/120/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='130_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/130/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='140_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/140/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=8
    lab=true
    save=false
    parallel=true
    mem=180""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='150_MLP'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","/scratch/slerman/XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/LATT/150/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    TestDataset=XRD.XRD
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
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
    # ['.*_No-Pool-CNN'],
    ['.*_MLP']
]

runs.XRD.title = 'LATT'
runs.XRD.sftp = True
