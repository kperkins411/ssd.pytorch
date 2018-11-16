import torch.optim as optim
import numpy as np
#logging for tensorflow
from tensorboardX import SummaryWriter

def do_requires_grad(model,*,requires_grad, apply_to_this_layer_on =0):
    '''
    Sets a model to be learnable or not
    :param model: network
    :param requires_grad:True gradient will be calculated, weights updated
    :param apply_to_this_layer_on will apply above to that layer forward
    :return:
    '''
    if model is None:
        raise ValueError("do_requires_grad() called with model==None")

    for i,param in enumerate(model.parameters()):
        if (i>=apply_to_this_layer_on):
            param.requires_grad = requires_grad

#from https://github.com/thomasjpfan/pytorch/blob/401ec389db2c9d2978917a6e4d1101b20340d7e7/torch/optim/lr_scheduler.py
#will eventually be pulled into pytorch repo
class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """
    TAG_CLR = "CLR"
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='exp_range', gamma=0.995,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.step_numb=0

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range', 'cosign_anneal'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular' :
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range'or self.mode == 'cosign_anneal':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'

        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        # self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration


    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)


    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
            self.step_numb+=1
        return lrs


class Writer:
    '''
    just a class to make it easy to track tensorboard events
    '''
    def __init__(self):
        self.writer = None
        self.last_batch_iteration = 0

    def __call__(self,name, lr):
        '''
        used to write to tensorboard
        :param name: string name of var to track
        :param lr:
        :return:
        '''
        if (self.writer is not None):
            self.writer.add_scalar(name, lr, self.last_batch_iteration)
            self.last_batch_iteration+=1

    def setWriter(self,writer):
        self.writer=writer


NUMB_IMAGES = 5011  # numb images in VOC2007
class CosAnnealLR(object):
    '''
       Part of stochastic gradient descent with warm restarts
       be careful with the learning rate, set it too high and you blow your model
       best to use some sort of learning rate finder

    '''
    RESTART_COSIGN_CYCLE = 0
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 batch_size = 64, numb_images= NUMB_IMAGES,epochs_per_cycle_schedule = [1,1,2,2,3] ):
        '''
        :param optimizer:
        :param base_lr:
        :param max_lr:
        :param batch_size:
        :param numb_images:
        :param epochs_per_cycle_schedule: this should sum to total number of epochs or your last epoch has lr somewhere
         between max_lr and base_lr
        '''

        #apply learning rates to optimizer layers
        self.optimizer = optimizer

        #how many iterations per epoch
        self.max_iter_per_cycle = numb_images // batch_size

        # if [1,2,3] then first cosign descent occurs over 1 epoch,
        # second over 2 epochs, 3rd over 3 epochs for a total of 6 epochs
        #any after that are over 3 epochs
        self.epochs_per_cycle_schedule = epochs_per_cycle_schedule
        self.epochs_per_cycle_schedule_loc = 0

        self.base_lr = base_lr
        self.max_lr = max_lr

        self.batch_size = batch_size
        self.numb_images = numb_images

        # covers the last partial batch
        self.pad = 1 if numb_images % self.batch_size > 0 else 0

        #start at 0
        self.current_iteration = CosAnnealLR.RESTART_COSIGN_CYCLE
        self.loc = CosAnnealLR.RESTART_COSIGN_CYCLE

        # self.scale_fn = self._exp_range_scale_fn    #OK to scale with this
        # self.scale_mode = 'iterations'
        self.writer = Writer()
        pass

    def _getNumberIterations(self):
        '''
        how many batches for this cycle
        if 1 epochs per cycle, then LRs contains numb batches   values from max_lr to base LR
        if 2 epochs per cycle, then LRs contains numb batches*2 values from max_lr to base LR
        :return:
        '''

        #how many epochs per cosign cycle
        multiplier = self.epochs_per_cycle_schedule[self.epochs_per_cycle_schedule_loc]
        self.epochs_per_cycle_schedule_loc+=1   #go to the next one

        return (self.numb_images//self.batch_size)*multiplier + self.pad

    def _getMaxLearningRate(self):
        '''
        reduce the max learning rate after each complete cosign cycle
        :return:
        '''
        self.max_lr_reducer=(self.max_lr - self.base_lr)/len(self.epochs_per_cycle_schedule)

        return self.max_lr- self.epochs_per_cycle_schedule_loc* self.max_lr_reducer

    def _calculate_learning_rate_range(self):
        '''
        generates a list of max_iter_per_cycle learning rates that starts at max_lr and descends
        to base_lr in a cosign wave
        max_iter_per_cycle changes according to schedule in epochs_per_cycle_schedule
        '''

        #gradually reduce the learning rate
        maxlr = self._getMaxLearningRate()

        # cosign varies between 1 and -1 (sweep of 2), factor to scale between lowlr and high_lr
        scale = 2 / (maxlr - self.base_lr)

        # then translate so ranges between low_lr and high_lr
        self.vals = (np.cos(np.linspace(0, np.pi, self._getNumberIterations())) / scale) + (maxlr - self.base_lr) / 2 + self.base_lr

    def _get_lr(self):
        # time to go to the next cycle length?
        if (self.loc == CosAnnealLR.RESTART_COSIGN_CYCLE):
            if (self.epochs_per_cycle_schedule_loc < len(self.epochs_per_cycle_schedule)):
                self._calculate_learning_rate_range()
                print(f"   number vals {len(self.vals)}")
            else:
                #just return the lowest learning rate
                self.loc=len(self.epochs_per_cycle_schedule)-1
                # raise (IndexError,"finished all epochs_per_cycle_schedule entries")

        lrs=[]
        lrs.append(self.vals[self.loc])

        self.writer('cosann', lrs[0])

        #set up for next time, either the next learning rate or restart
        self.loc +=1
        if (self.loc ==len(self.vals)):
            self.loc = CosAnnealLR.RESTART_COSIGN_CYCLE

        return lrs

    def batch_step(self):
         for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr

    def setWriter(self, writer):
         '''
         for tensorboard
         :param writer:
         :return:
         '''
         self.writer.setWriter(writer)

# class CosAnnealLR(CyclicLR):
#     '''
#     Part of stochastic gradient descent with warm restarts
#     be careful with the learning rate, set it too high and you blow your model
#     best to use some sort of learning rate finder
#
#     '''
#
#     RESTART_COSIGN_CYCLE = -1
#
#     def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
#                  mode='cos_anneal', gamma=1,
#                  scale_fn=None, scale_mode='cycle', last_batch_iteration=-1, batch_size = 64, numb_images= NUMB_IMAGES,epochs_per_cycle_schedule = [1,1,2,2,3] ):
#         super().__init__(optimizer, base_lr, max_lr,
#                  mode, gamma,
#                  scale_fn, scale_mode, last_batch_iteration)
#
#         #how many iterations per epoch
#         self.max_iter_per_cycle = numb_images // batch_size
#
#         # if [1,2,3] then first cosign descent occurs over 1 epoch,
#         # second over 2 epochs, 3rd over 3 epochs for a total of 6 epochs
#         #any after that are over 3 epochs
#         self.epochs_per_cycle_schedule = epochs_per_cycle_schedule
#         self.epochs_per_cycle_schedule_loc = 0
#
#         self.base_lr = base_lr
#         self.max_lr = max_lr
#
#         #diff between base_lr and max_lr
#         diff = max_lr - base_lr
#         total_epochs = sum(self.epochs_per_cycle_schedule)
#         self.max_lr_reducer = diff/total_epochs
#
#         #add to max_lr so it all works out on first calculation
#         self.max_lr +=self.max_lr_reducer
#
#         #start at 0
#         self.current_iteration = CosAnnealLR.RESTART_COSIGN_CYCLE
#         self.loc = CosAnnealLR.RESTART_COSIGN_CYCLE
#
#         self.scale_fn = self._exp_range_scale_fn    #OK to scale with this
#         self.scale_mode = 'iterations'
#         self.writer = SummaryWriter()
#
#     def calculate_learning_rate_range(self):
#         '''
#         generates a list of max_iter_per_cycle learning rates that starts at max_lr and descends
#         to base_lr in a cosign wave
#
#         max_iter_per_cycle changes according to schedule in epochs_per_cycle_schedule
#         '''
#         #gradually reduce the learning rate
#         self.max_lr = self.max_lr-self.max_lr_reducer
#
#         # cosign varies between 1 and -1 (sweep of 2), factor to scale between lowlr and high_lr
#         scale = 2 / (self.max_lr - self.base_lr)
#
#         # then translate so ranges between low_lr and high_lr
#         self.vals = (np.cos(np.linspace(0, np.pi, self.max_iter_per_cycle)) / scale) + (self.max_lr - self.base_lr) / 2 + self.base_lr
#
#     def setWriter(self,writer):
#         self.writer = writer
#
#     # def __getCycleMult():
#     #
#
#     def get_lr(self):
#
#         # time to go to the next cycle length?
#         if (self.loc == CosAnnealLR.RESTART_COSIGN_CYCLE):
#             if (self.epochs_per_cycle_schedule_loc < len(self.epochs_per_cycle_schedule)):
#                 self.max_iter_per_cycle= self.max_iter_per_cycle * self.epochs_per_cycle_schedule[self.epochs_per_cycle_schedule_loc]
#                 self.epochs_per_cycle_schedule_loc += 1
#                 self.calculate_learning_rate_range()
#                 print(f"   number vals {len(self.vals)}")
#                 self.current_iteration = CosAnnealLR.RESTART_COSIGN_CYCLE
#
#         lrs=[]
#         lrs.append(self.vals[self.loc])
#
#         self.writer.add_scalar('cosann', lrs[0], self.last_batch_iteration)
#
#         #set up for next time
#         self.current_iteration += 1
#         self.loc= (self.current_iteration)%self.max_iter_per_cycle
#
#         return lrs



