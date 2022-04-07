from os import path

import librosa as rosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from config import CONFIG

from utils.stft import STFTMag

matplotlib.use('Agg')


class TensorBoardLoggerExpanded(TensorBoardLogger):
    def __init__(self, sr=16000):
        super().__init__(save_dir='lightning_logs', default_hp_metric=False, name='')
        self.sr = sr
        self.stftmag = STFTMag()

    def fig2np(self, fig):
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def plot_spectrogram_to_numpy(self, y, step, name):
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle(f'Epoch_{step}')
        ax = plt.subplot(1, 1, 1)
        ax.set_title(name)

        plt.imshow(rosa.amplitude_to_db(self.stftmag(y).numpy(),
                                        ref=np.max, top_db=80.),
                   # vmin = -20,
                   vmax=0.,
                   aspect='auto',
                   origin='lower',
                   interpolation='none')
        plt.colorbar()
        plt.xlabel('Frames')
        plt.ylabel('Channels')
        plt.tight_layout()

        fig.canvas.draw()
        data = self.fig2np(fig)

        plt.close()
        return data

    @rank_zero_only
    def log_spectrogram(self, y, y_low, y_recon, epoch):
        y, y_low, y_recon = y.detach().cpu(), y_low.detach().cpu(), y_recon.detach().cpu()
        name_list = ['y', 'y_low', 'y_recon']
        for i, audio in enumerate([y, y_low, y_recon]):
            spec_img = self.plot_spectrogram_to_numpy(audio, epoch, name_list[i])
            print("Logging image", name_list[i])
            self.experiment.add_image(path.join(self.save_dir, 'result', name_list[i]),
                                  spec_img,
                                  epoch,
                                  dataformats='HWC')
        self.experiment.flush()
        return

    @rank_zero_only
    def log_inference_audio(self, audio, tag):
        print("Logging inference audio", audio.shape)
        self.experiment.add_audio(path.join(self.save_dir, 'inference', 'audio_result', tag),
                              audio,
                              sample_rate=CONFIG.DATA.sr,
                              )
        self.experiment.flush()
        return

    @rank_zero_only
    def log_validation_audio(self, y, y_low, y_recon, epoch):
        name_list = ['y', 'y_low', 'y_recon']
        for i, audio in enumerate([y, y_low, y_recon]):
            print("Logging validation audio", name_list[i], audio.shape)
            self.experiment.add_audio(path.join(self.save_dir, 'validation', 'audio_result', name_list[i]),
                                  audio,
                                  epoch,
                                  sample_rate=CONFIG.DATA.sr
                                  )
        self.experiment.flush()
        return
