from Sweeps.Templates import template


runs = template('XRD')

runs.XRD.sweep = [
    # # Mix-Soup, No-Pool-CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.NoPoolCNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Mix-Soup_${dataset.num_classes}-Way'
    # experiment='No-Pool-CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_mix/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0.5]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # parallel=true
    # num_gpus=8
    # mem=180""",

    # # Large + RRUFF, No-Pool-CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.NoPoolCNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Large-and-RRUFF_${dataset.num_classes}-Way'
    # experiment='No-Pool-CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # parallel=true
    # num_gpus=8
    # mem=180""",
    #
    # # Mix + RRUFF, No-Pool-CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.NoPoolCNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Mix-and-RRUFF_${dataset.num_classes}-Way'
    # experiment='No-Pool-CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_mix/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # parallel=true
    # num_gpus=8
    # mem=180""",
    #
    # # Large + RRUFF, CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.NoPoolCNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Large-and-RRUFF_${dataset.num_classes}-Way'
    # experiment='CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # parallel=true
    # num_gpus=8
    # mem=180""",
    #
    # # Mix + RRUFF, CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.CNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Mix-and-RRUFF_${dataset.num_classes}-Way'
    # experiment='CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_mix/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # parallel=true
    # num_gpus=8
    # mem=180""",

    # # Mix-Soup, CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.CNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Mix-Soup_${dataset.num_classes}-Way'
    # experiment='CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_mix/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0.5]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # num_workers=1
    # parallel=true
    # num_gpus=8
    # mem=180""",
    #
    # # Mix-Soup, MLP
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=Identity
    # Predictor=XRD.MLP
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Mix-Soup_${dataset.num_classes}-Way'
    # experiment='MLP'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_mix/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0.5]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # num_workers=1
    # parallel=true
    # num_gpus=8
    # mem=180""",

    # Large-Soup, No-Pool-CNN
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
    # parallel=true
    # num_gpus=8
    # mem=180""",

    # # Large-Soup, CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.CNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Large-Soup_${dataset.num_classes}-Way'
    # experiment='CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0.5]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # num_workers=1
    # parallel=true
    # num_gpus=8
    # mem=180""",
    #
    # # Large-Soup, MLP
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=Identity
    # Predictor=XRD.MLP
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Large-Soup_${dataset.num_classes}-Way'
    # experiment='MLP'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0.5]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # num_workers=1
    # parallel=true
    # num_gpus=8
    # mem=180""",

    # # Mix, No-Pool-CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.NoPoolCNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Mix_${dataset.num_classes}-Way'
    # experiment='No-Pool-CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_mix/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # num_workers=1
    # parallel=true
    # num_gpus=8
    # mem=180""",

    # # Mix, CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.CNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Mix_${dataset.num_classes}-Way'
    # experiment='CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_mix/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # num_workers=1
    # parallel=true
    # num_gpus=8
    # mem=180""",
    #
    # # Mix, MLP
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=Identity
    # Predictor=XRD.MLP
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Mix_${dataset.num_classes}-Way'
    # experiment='MLP'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_mix/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # num_workers=1
    # parallel=true
    # num_gpus=8
    # mem=180""",

    # # Large, No-Pool-CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.NoPoolCNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Large_${dataset.num_classes}-Way'
    # experiment='No-Pool-CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # num_workers=1
    # parallel=true
    # num_gpus=8
    # mem=180""",

    # # Large, CNN
    # """task=classify/custom
    # Dataset=XRD.XRD
    # Aug=Identity
    # Trunk=Identity
    # Eyes=XRD.CNN
    # Predictor=XRD.Predictor
    # batch_size=256
    # standardize=false
    # norm=true
    # task_name='Large_${dataset.num_classes}-Way'
    # experiment='CNN'
    # '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    # +'dataset.train_eval_splits=[1, 0]'
    # +dataset.num_classes=7,230
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # num_workers=1
    # parallel=true
    # num_gpus=8
    # mem=180""",

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
    # train_steps=5e5
    # save=true
    # logger.wandb=true
    # lab=true
    # num_workers=1
    # parallel=true
    # num_gpus=8
    # mem=180""",

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
    # num_gpus=8
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
    # num_gpus=8
    # mem=20""",
]

# Create universal replays - Note: define for Mix-Soup and Mix+RRUFF
# runs.XRD.sweep = [
#     # Mix, MLP
#     """task=classify/custom
#     Dataset=XRD.XRD
#     Aug=Identity
#     Trunk=Identity
#     Eyes=Identity
#     Predictor=XRD.MLP
#     batch_size=256
#     standardize=false
#     norm=true
#     task_name='Mix_${dataset.num_classes}-Way'
#     experiment='MLP'
#     '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_mix/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
#     +'dataset.train_eval_splits=[1, 0]'
#     +dataset.num_classes=230
#     train_steps=1
#     save=false
#     logger.wandb=true
#     lab=true
#     num_workers=1
#     num_gpus=1
#     mem=180""",

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

# Generalization To Magnetic Properties
runs.XRD.sweep = [
    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='No-Pool-CNN_icsd'
    'dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/mp_icsd_shen/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=1
    lab=true
    save=false
    mem=5""",

    # Large-Soup, CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='CNN_icsd'
    'dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/mp_icsd_shen/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=1
    lab=true
    save=false
    mem=5""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='MLP_icsd'
    'dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/mp_icsd_shen/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=1
    lab=true
    save=false
    mem=5""",

    # Mix-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Mix-Soup_${test_dataset.num_classes}-Way'
    experiment='No-Pool-CNN_icsd'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_mix/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/mp_icsd_shen/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Mix-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=1
    lab=true
    save=false
    mem=5""",

    # Large-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='No-Pool-CNN_nonicsd'
    'dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/mp_nonicsd_shen/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=1
    lab=true
    save=false
    mem=5""",

    # Large-Soup, CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='CNN_nonicsd'
    'dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/mp_nonicsd_shen/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/CNN/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=1
    lab=true
    save=false
    mem=5""",

    # Large-Soup, MLP - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Large-Soup_${test_dataset.num_classes}-Way'
    experiment='MLP_nonicsd'
    'dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/mp_nonicsd_shen/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/MLP/DQNAgent/classify/Large-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=1
    lab=true
    save=false
    mem=5""",

    # Mix-Soup, No-Pool-CNN - Generalize To MP
    """task=classify/custom
    Dataset=XRD.XRD
    batch_size=256
    task_name='Mix-Soup_${test_dataset.num_classes}-Way'
    experiment='No-Pool-CNN_nonicsd'
    '+dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd171k_mix/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    '+test_dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/mp_nonicsd_shen/"]'
    +'test_dataset.train_eval_splits=[0]'
    +test_dataset.num_classes=7,230
    train_steps=0
    load=true
    load_path='/scratch/slerman/UnifiedML/Checkpoints/No-Pool-CNN/DQNAgent/classify/Mix-Soup_${test_dataset.num_classes}-Way_1.pt'
    logger.wandb=true
    num_workers=8
    num_gpus=1
    lab=true
    save=false
    mem=5""",
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
#     'PS1.*', 'Large.*', 'Mix.*'
# ]
#
# runs.XRD.title = 'RRUFF'

runs.XRD.plots = [
    ['CNN', 'MLP', 'No-Pool-CNN'],
]

runs.XRD.tasks = [
    '.*on_MP.*',
]

runs.XRD.title = 'Magnetic Properties'
