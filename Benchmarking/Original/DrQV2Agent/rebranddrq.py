import glob
import pandas as pd


def rename():
    files = glob.glob('./**/*.csv', recursive=True)

    for file in files:

        csv = pd.read_csv(file)
        csv.reset_index(drop=True, inplace=True)

        file = file.replace('dmc_', '').replace('.csv', '_10_Eval.csv')

        csv['step'] = csv['frame'] / 2
        csv['reward'] = csv['episode_reward']
        csv['time'] = csv['hour']
        csv['seed'] = csv['seed']

        csv = csv[['seed', 'time', 'step', 'frame', 'reward']]

        csv.to_csv(file, index=False)


if __name__ == "__main__":
    rename()
