runs = {'XRD': {
    'sweep': [
        # Large-Soup, 50-50, ViT, 7/230-Way, noise 0 - Launched ✓ (Cornea)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=ViT
        Predictor=XRD.Predictor
        batch_size=256
        standardize=false
        norm=true
        task_name='Large-Soup-50-50_${dataset.num_classes}-Way'
        experiment='ViT_optim_ADAM_batch_size_${batch_size}'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0.5]'
        +dataset.num_classes=7,230
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Large-Soup, 50-50, No-Pool-CNN, 7/230-Way, noise 0 - Launched ✓ (Cornea)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=XRD.NoPoolCNN
        Predictor=XRD.Predictor
        batch_size=256
        standardize=false
        norm=true
        task_name='Large-Soup-50-50_${dataset.num_classes}-Way'
        experiment='No-Pool-CNN_optim_ADAM_batch_size_${batch_size}'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0.5]'
        +dataset.num_classes=7,230
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Large-Soup, 50-50, CNN, 7/230-Way, noise 0 - Launched ✓ (Cornea)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=XRD.CNN
        Predictor=XRD.Predictor
        batch_size=256
        standardize=false
        norm=true
        task_name='Large-Soup-50-50_${dataset.num_classes}-Way'
        experiment='CNN_optim_ADAM_batch_size_${batch_size}'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0.5]'
        +dataset.num_classes=7,230
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Large-Soup, 50-50, MLP, 7/230-Way, noise 0 - Launched ✓ (Cornea)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=Identity
        Predictor=XRD.MLP
        batch_size=256
        standardize=false
        norm=true
        task_name='Large-Soup-50-50_${dataset.num_classes}-Way'
        experiment='MLP_optim_ADAM_batch_size_${batch_size}'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0.5]'
        +dataset.num_classes=7,230
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Soup, 50-50, CNN, 7/230-Way, noise 0 - Launched ✓ (Cornea)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=XRD.CNN
        Predictor=XRD.Predictor
        batch_size=256,32
        standardize=false
        norm=false
        task_name='Soup-50-50_${dataset.num_classes}-Way'
        experiment='CNN_optim_ADAM_batch_size_${batch_size}'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0.5]'
        +dataset.num_classes=7,230
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Soup, 50-50, CNN, 7-Way, noise 0, SGD, LR 1e-3, Batch Size 256 - Launched ✓ (Cornea)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=XRD.CNN
        Predictor=XRD.Predictor
        batch_size=256
        Optim=SGD
        lr=1e-3
        standardize=false
        norm=false
        task_name='Soup-50-50_${dataset.num_classes}-Way'
        experiment='CNN_optim_SGD_batch_size_${batch_size}_lr_1e-3'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0.5]'
        +dataset.num_classes=7
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Soup, 50-50, MLP, 7/230-Way, noise 0 - Launched ✓ (Cornea)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=Identity
        Predictor=XRD.MLP
        batch_size=32,256
        standardize=false
        norm=false
        task_name='Soup-50-50_${dataset.num_classes}-Way'
        experiment='MLP_optim_ADAM_batch_size_${batch_size}'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0.5]'
        +dataset.num_classes=7,230
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Soup, 50-50, MLP, 7-Way, SGD, LR 1e-3, Batch Size 256 - Launched ✓ (Cornea)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=Identity
        Predictor=XRD.MLP
        batch_size=256
        standardize=false
        norm=false
        lr=1e-3
        Optim=SGD
        task_name='Soup-50-50_${dataset.num_classes}-Way'
        experiment='MLP_optim_SGD_batch_size_${batch_size}_lr_1e-3'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0.5]'
        +dataset.num_classes=7
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Soup, 50-50, CNN, 7/230-Way, noise 0, norm - Launched ✓ (Cornea)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=XRD.CNN
        Predictor=XRD.Predictor
        batch_size=256
        standardize=false
        norm=true
        task_name='Soup-50-50_${dataset.num_classes}-Way'
        experiment='CNN_optim_ADAM_batch_size_${batch_size}_norm'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0.5]'
        +dataset.num_classes=7,230
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Soup, 50-50, CNN, 7/230-Way, noise 0, standardized - Launched ✓ (Cornea)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=XRD.CNN
        Predictor=XRD.Predictor
        batch_size=256
        standardize=true
        norm=false
        task_name='Soup-50-50_${dataset.num_classes}-Way'
        experiment='CNN_optim_ADAM_batch_size_${batch_size}_standardized'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0.5]'
        +dataset.num_classes=7,230
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Soup, 50-50, CNN, 7-Way, noise 2 - Launched ✓ (Macula)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Blocks.Augmentations.IntensityAug
        +aug.noise=2
        Trunk=Identity
        Eyes=XRD.CNN
        Predictor=XRD.Predictor
        batch_size=256
        standardize=false
        norm=false
        task_name='Soup-50-50_${dataset.num_classes}-Way'
        experiment='CNN_optim_ADAM_batch_size_${batch_size}_noise_2'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0.5]'
        +dataset.num_classes=7
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Soup, 50-50, ResNet18, 7-Way, noise 0 - Launched X (Macula) - Too slow!
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=ResNet18
        Predictor=XRD.Predictor
        batch_size=256
        standardize=false
        norm=false
        task_name='Soup-50-50_${dataset.num_classes}-Way'
        experiment='ResNet18_optim_ADAM_batch_size_${batch_size}'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0.5]'
        +dataset.num_classes=7
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Synthetic-Only, CNN, 7-Way, noise 0 - Launched ✓ (Cornea)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=XRD.CNN
        Predictor=XRD.Predictor
        batch_size=256
        standardize=false
        norm=false
        task_name='Synthetic_${dataset.num_classes}-Way'
        experiment='CNN_optim_ADAM_batch_size_${batch_size}'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0]'
        +dataset.num_classes=7
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Synthetic-Only, MLP, 7-Way, noise 0 - Launched ✓ (Macula)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=Identity
        Predictor=XRD.MLP
        batch_size=256
        standardize=false
        norm=false
        task_name='Synthetic_${dataset.num_classes}-Way'
        experiment='MLP_optim_ADAM_batch_size_${batch_size}'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0]'
        +dataset.num_classes=7
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Synthetic-Only, CNN, 7-Way, noise 0
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=XRD.CNN
        Predictor=XRD.Predictor
        batch_size=256
        standardize=false
        norm=false
        Optim=SGD
        lr=1e-3
        task_name='Synthetic_${dataset.num_classes}-Way'
        experiment='CNN_optim_SGD_batch_size_${batch_size}_lr_1e-3'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0]'
        +dataset.num_classes=7
        train_steps=5e5
        save=true
        logger.wandb=true""",

        # Synthetic-Only, MLP, 7-Way, noise 0 - Launched ✓ (Cornea)
        """python Run.py
        task=classify/custom
        Dataset=XRD.XRD
        Aug=Identity
        Trunk=Identity
        Eyes=Identity
        Predictor=XRD.MLP
        batch_size=256
        standardize=false
        norm=false
        Optim=SGD
        lr=1e-3
        task_name='Synthetic_${dataset.num_classes}-Way'
        experiment='MLP_optim_SGD_batch_size_${batch_size}_lr_1e-3'
        '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
        +'dataset.train_eval_splits=[1, 0]'
        +dataset.num_classes=7
        train_steps=5e5
        save=true
        logger.wandb=true""",
    ],
    'plots': [
        # Summary bar & line plots  TODO rename Large-Pool -> "..._norm"
        ['MLP_optim_ADAM_batch_size_256'],
        # Summary bar & line plots  TODO rename Large-Pool -> "..._norm"
        # ['.*CNN_optim_ADAM_batch_size_256',
        #  '.*MLP_optim_ADAM_batch_size_256_norm',
        #  'MLP_optim_ADAM_batch_size_256.*'],
    ],
    'sftp': True,
    'bluehive': False,
    'lab': True,
    'steps': 5e5,
    'title': 'RRUFF',
    'x_axis': 'Step',
    'bluehive_only': [],
    'tasks': ['.*7-Way.*'],
    'agents': [],
    'suites': []},
}
