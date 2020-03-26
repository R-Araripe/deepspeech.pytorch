import os

import torch
import torchvision


def to_np(x):
    return x.cpu().numpy()


class VisdomLogger(object):
    def __init__(self, id, num_epochs):
        from visdom import Visdom
        self.viz = Visdom()
        self.opts = dict(title=id, ylabel='', xlabel='Epoch', legend=['Loss', 'WER', 'CER'])
        self.viz_window = None
        self.epochs = torch.arange(1, num_epochs + 1)
        self.visdom_plotter = True

    def update(self, epoch, values):
        x_axis = self.epochs[0:epoch + 1]
        y_axis = torch.stack((values["loss_results"][:epoch + 1],
                              values["wer_results"][:epoch + 1],
                              values["cer_results"][:epoch + 1]),
                             dim=1)
        self.viz_window = self.viz.line(
            X=x_axis,
            Y=y_axis,
            opts=self.opts,
            win=self.viz_window,
            update='replace' if self.viz_window else None
        )

    def load_previous_values(self, start_epoch, package):
        self.update(start_epoch - 1, package)  # Add all values except the iteration we're starting from


class TensorBoardLogger(object):
    def __init__(self, id, log_dir, log_params, parameters=None, comment=''):
        # import pdb; pdb.set_trace()

        log_dir_name = os.path.join(log_dir, id, comment)
        os.makedirs(log_dir_name, exist_ok=True)

        # from tensorboardX import SummaryWriter
        from torch.utils.tensorboard import SummaryWriter

        self.id = id
        # TO DO get parameters and transform to string to identify run
        self.tensorboard_writer = SummaryWriter(log_dir_name, flush_secs=10)
        self.log_params = log_params

    def update(self, iter, values, parameters=None, debug=False, together=False, name=''):
        '''
            Parameters
            ----------

            values: dicionary where key is metric name and value..the value

        '''
        if debug:
            import pdb; pdb.set_trace()

        if together:
            self.tensorboard_writer.add_scalars(name, values, iter)
        for k, v in values.items():
            self.tensorboard_writer.add_scalar(k, v, iter)

        if self.log_params and parameters is not None:
            for tag, value in parameters():
                tag = tag.replace('.', '/')
                # import pdb; pdb.set_trace()
                self.tensorboard_writer.add_histogram(tag, value, iter)
                if value.requires_grad:
                    self.tensorboard_writer.add_histogram(tag + '/grad', value.grad, iter)


    def load_previous_values_libri(self, start_epoch, values):
        # import pdb; pdb.set_trace()
        loss_results = values["loss_results"][:start_epoch]
        wer_results = values["wer_results"][:start_epoch]
        cer_results = values["cer_results"][:start_epoch]

        for i in range(start_epoch):
            values = {
                'Avg Train Loss pre-trained': loss_results[i],
                'Avg WER pre-trained': wer_results[i],
                'Avg CER pre-trained': cer_results[i]
            }
            self.tensorboard_writer.add_scalars(self.id, values, i + 1)

    def close(self):
        self.tensorboard_writer.close()

    def add_image(self, images, sizes, labels, network=None):
        grid = torchvision.utils.make_grid(images)
        # self.tensorboard_writer.add_image('train_random_batch_images', grid)
        # import pdb; pdb.set_trace()
        if network:
            self.tensorboard_writer.add_graph(network, (images, sizes), verbose=False)
