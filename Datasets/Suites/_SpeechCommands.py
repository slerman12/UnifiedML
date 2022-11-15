import torch

from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.classes = sorted(list(set(exp[2] for exp in self)))  # 'marvin', 'visual', 'zero', ... etc
        print(self.classes)

    def collate_fn(self, batch):
        waveform, label = zip(*batch)
        return pad_sequence(waveform), torch.stack(label)  # Pad waveform

    def __getitem__(self, item):
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(item)

        # Can resample
        # new_sample_rate = 8000
        # transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        # waveform = transform(waveform)

        return waveform, torch.tensor(self.classes.index(label))


def pad_sequence(batch: (list, tuple)):
    # Pad uneven sequences in a list with zeros up to the same max length
    return torch.nn.utils.rnn.pad_sequence([item.t() for item in batch],
                                           batch_first=True, padding_value=0.).permute(0, 2, 1)
