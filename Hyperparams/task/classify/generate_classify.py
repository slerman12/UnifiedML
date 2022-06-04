IMAGE_DATASETS = [
    'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
    'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
    'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
    'VOCSegmentation', 'VOCDetection', 'Cityscapes', 'ImageNet',
    'Caltech101', 'Caltech256', 'CelebA', 'WIDERFace', 'SBDataset',
    'USPS', 'Kinetics400', "Kinetics", 'HMDB51', 'UCF101',
    'Places365', 'Kitti', "INaturalist", "LFWPeople", "LFWPairs",
    'TinyImageNet', 'Custom'
]

if __name__ == '__main__':
    out = ""
    for task in IMAGE_DATASETS:
        f = open(f"./{task.lower()}.yaml", "w")
        f.write(r"""defaults:
      - _self_
    
suite: classify
train_steps: 200000
stddev_schedule: 'linear(1.0,0.1,100000)'
frame_stack: null
action_repeat: null
nstep: 0
evaluate_per_steps: 1000
evaluate_episodes: 1
learn_per_steps: 1
learn_steps_after: 0
seed_steps: 50
explore_steps: 0
log_per_episodes: 10
offline: true
task_name: {}""".format(task if task != 'Custom' else task + f'{".${Dataset}"}'))
        f.close()
        out += ' "' + task.lower() + '"'
    print(out)
