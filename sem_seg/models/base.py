from pathlib import Path
from typing import List, Callable, Dict
from tqdm import tqdm
import time
import copy
import torch
from torch.utils.tensorboard import SummaryWriter

from . import eval_metrices
from training import utils

DIRNAME = Path(__file__).parents[1].resolve()
WEIGHTSDIR = DIRNAME / "weights"
LOGSDIR = DIRNAME / "tensorboard_logs"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model:
    """ Base Class for training and evaluation of any network """

    def __init__(self, network: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim,
                 criterion: Callable,
                 lr_scheduler: torch.optim.lr_scheduler,
                 additional_identifier: str = ''):
        self.network = network
        self.dataloader = dataloader
        self.optim = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.name = f"{self.__class__.__name__}_{self.network.__class__.__name__}"
        if additional_identifier:
            self.name += '_' + additional_identifier
        self.metrices = ['accuracy']
        self.logToTensorboard = True
        self.writer = None

    @property
    def weights_file_name(self) -> str:
        """
        :return: returns the path where to store/load the weights of the model.
        """
        WEIGHTSDIR.mkdir(parents=True, exist_ok=True)
        return str(WEIGHTSDIR / f"{self.name}.h5")

    def train(self, num_epochs: int) -> None:
        """
        1. set up the model in appropriate training mode(Use this to setup DDP model for distributed training also if
        required)
        2. Train the model for num_epochs and keep track of the best_model_weights based on self.metric[0] values.
        3. Log the epoch results( loss and other stats mentioned in self.metrices) on tensorboard
        :param num_epochs: Number of training epochs
        :return: None
        """
        since = time.time()
        losses = {'train': [], 'val': []}
        stats = {'train': [], 'val': []}
        lr_rates = []
        best_epoch, self.best_stat = None, float('-inf')
        best_model_weights = None
        self.network.to(device)

        for epoch in tqdm(range(num_epochs)):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.network.train()
                else:
                    self.network.eval()
                running_loss, running_stats = 0.0, [0.0] * len(self.metrices)

                for inputs, outputs in self.dataloader[phase]:
                    inputs, outputs = inputs.to(device), outputs.to(device)
                    self.optim.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        network_outputs = self.network(inputs)
                        loss = self.criterion(network_outputs, outputs)
                        batch_stats = self.evaluate_metrices(network_outputs, outputs)

                        if phase == 'train':
                            loss.backward()
                            self.optim.step()

                    n = inputs.size(0)
                    running_loss += loss.item() * n
                    running_stats = self.update_running_stats(batch_stats, running_stats, n)
                N = len(self.dataloader[phase].dataset)
                epoch_loss = running_loss / N
                epoch_stats = self.get_epoch_stats(running_stats, N)

                if phase == 'val' and epoch_stats[0] > self.best_stat:
                    self.best_stat = epoch_stats[0]
                    best_epoch = epoch
                    best_model_weights = copy.deepcopy(self.network.state_dict())
                losses[phase].append(epoch_loss)
                stats[phase].append(epoch_stats)

            self.log_stats(epoch, losses, stats, [inputs, network_outputs, outputs], lr_rates)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_stats[0])  # considering ReduceLrOnPlateau and watch val mIOU
                lr_step = self.optim.state_dict()["param_groups"][0]["lr"]
                lr_rates.append(lr_step)
                min_lr = self.lr_scheduler.min_lrs[0]
                if lr_step / min_lr <= 10:
                    print("Min LR reached")
                    break

        time_elapsed = time.time() - since
        print('Training Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best {}: {:.6f} and best epoch is {}'.format(self.metrices[0], self.best_stat, best_epoch + 1))
        self.network.load_state_dict(best_model_weights)

    def evaluate_metrices(self, outs: torch.Tensor, targets: torch.Tensor) -> List:
        """
        :param outs: output logits from network for a given batch of inputs
        :param targets: ground truth targets corresponding to the batch of inputs
        :return: calculated metric from the utility function
        """
        fn_mapping = {
            'pixel_accuracy': eval_metrices.pixel_accuracy,
            'mIOU': eval_metrices.get_confusion_matrix,
            'accuracy': eval_metrices.accuracy
        }
        scores = []
        for mat in self.metrices:
            scores.append(fn_mapping[mat](outs, targets))
        return scores

    def update_running_stats(self, batch_stats: List, running_stats: List, num_inputs_in_batch: int) -> List:
        """
        :param batch_stats: List of batch stats(in the same order as self.metrices)
        :param running_stats: List of running stats from the current epoch so far(in the same order as self.metrices)
        :param num_inputs_in_batch: Number of inputs in current batch
        :return: Aggregated stats for the current epoch. MIOU requires the bins_matrix(CxC) to be summed up.
        """
        running_stats = [r_stat + b_stat * num_inputs_in_batch if 'miou' not in key.lower() else r_stat + b_stat
                         for key, r_stat, b_stat in zip(self.metrices, running_stats, batch_stats)]
        return running_stats

    def get_epoch_stats(self, running_stats: List, total_inputs: int) -> List:
        epoch_stats = [stat / total_inputs if 'miou' not in key.lower() else eval_metrices.mIOU(stat)
                       for key, stat in zip(self.metrices, running_stats)]
        return epoch_stats

    def log_stats(self, epoch: int, losses: List[Dict], stats: List[Dict] = [], data: List = [], lr_rates: List = []):
        """ Format epoch stats and log them to console and to tensorboard if enabled"""
        if epoch == 0:
            metrices_title = '\t'.join(
                [(phase + '_' + mat)[:12] for mat in self.metrices for phase in ['train', 'val']])
            tqdm.write('Epoch\tTrain_Loss\tVal_Loss\t' + metrices_title)
            tqdm.write(
                '-----------------------------------------------------------------------------------------------------')
        metrices_stats = ' \t '.join([format(stats[phase][epoch][i], '.6f') for i in range(len(self.metrices))
                                      for phase in ['train', 'val']])
        log_stats = ' \t '.join([format(losses[phase][epoch], '.6f') for phase in ['train', 'val']])
        tqdm.write('{:4d} \t'.format(epoch + 1) + log_stats + '\t' + metrices_stats)
        if self.logToTensorboard:
            self.writeToTensorboard(epoch, losses, stats, data, lr_rates)

    def writeToTensorboard(self, epoch: int, losses: List[Dict], stats: List[Dict] = [], data: List = [],
                           lr_rates: List = []):
        """Logs epoch stats to tensorboard and output predictions to tensorboard after every 25 epoch"""
        if self.writer is None:
            self.setup_tensorboard()
        self.writer.add_scalar('Loss/train', losses['train'][epoch], epoch + 1)
        self.writer.add_scalar('Loss/val', losses['val'][epoch], epoch + 1)
        num_batches = len(lr_rates)
        for i, lr in enumerate(lr_rates):
            self.writer.add_scalar('Loss/schedule_lr', lr, epoch * num_batches + 1)
        if epoch % 25 != 0:
            return
        for i, metric in enumerate(self.metrices):
            self.writer.add_scalar(metric + '/train', stats['train'][epoch], epoch + 1)
            self.writer.add_scalar(metric + '/val', stats['val'][epoch], epoch + 1)
        if data:
            imgs, network_outputs, targets = data
            _, preds = torch.max(network_outputs, dim=1)
            val_results = utils.plot_images(imgs[:4], targets[:4], preds[:4],
                                            title='val_mIOU' + str(stats['val'][epoch][0]),
                                            num_cols=3)
            self.writer.add_figure('val_epoch' + str(epoch), val_results)

    def add_text_to_tensorboard(self, text: str) -> None:
        """ Log text to tensorboard. Used for logging experiment configuration"""
        if self.writer is None:
            self.setup_tensorboard()
        self.writer.add_text("model_details", "Experiment Config " + text)

    def setup_tensorboard(self):
        """ setup the tensorboard logging dir"""
        logsdir = LOGSDIR / f"{self.name}"
        logsdir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(logsdir)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.Tensor [N, C, H, W] where C is number of channels in inputs
        :return: logit Tensor of size [N, K, H, W] where K is number of output classes
        """
        self.network.eval()
        if not x.is_cuda:
            x = x.to(device)
            self.network.to(device)
        outs = self.network(x)
        return outs

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input Tensor [N, C, H, W]
        :return: Predict Tensor [N, H, W]
        """
        outs = self.get_logits(x)
        _, preds = torch.max(outs, dim=1)
        return preds

    def load_weights(self, filename=None) -> None:
        """load model weights from passed filename or self.weights_file_name"""
        if filename is not None:
            filename = self.weights_file_name
        self.network.load_state_dict(torch.load(filename))

    def save_weights(self) -> None:
        """ Save current model state dict int self.weights_file_name """
        torch.save(self.network.state_dict(), self.weights_file_name)
