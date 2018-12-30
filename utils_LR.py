import torch.optim as optim
import numpy as np
import math

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

class LR(object):
    '''
    Base class, returns a single cycle of learning rates
    '''
    def __call__(self, *args, **kwargs):
        return self.getLRs()

    def getLRs(self,numb_iterations,max_lr, min_lr):
        '''
        :param numb_iterations: number total samples
        :param max_lr: upper
        :param min_lr: lower
        :return: list of learning rates, numb_iterations long
        '''
        raise NotImplementedError

class TriangularLR(LR):
    def getStepSize(self,numb_iterations):
        return numb_iterations // 2

    def getLRs(self, numb_iterations, max_lr, min_lr):
        # determines the halfway point
        step_size = self.getStepSize(numb_iterations)
        data = []
        for itr in range(numb_iterations):
            cycle = math.floor(1 + itr / (2 * step_size))
            x = abs(itr / step_size - 2 * cycle + 1)
            lr = min_lr + (max_lr - min_lr) * max(0, (1 - x))
            data.append(lr)
        return data

class TriangularLR_LRFinder(TriangularLR):
    def getStepSize(self,numb_iterations):
        return numb_iterations #makes stepsize and numb_iterations equal, you get first half of triangular wave

class CosignLR(LR):
    def getLRs(self, numb_iterations, max_lr, min_lr):
        '''
        Cosign that starts at max_lr and decreases to min_lr
        '''

        # then translate so ranges between low_lr and high_lr
        data = (np.cos(np.linspace(0, np.pi, numb_iterations))) + (max_lr - min_lr) / 2 + min_lr
        return data

class LR_anneal(object):
     def getMaxLR(self,step_size,max_lr,min_lr):
        raise NotImplemented

class LR_anneal_linear(LR_anneal):
    def __init__(self):
        self.cur_decrement = 0                 #ensures start at 0

    def getMaxLR(self,step_size,max_lr,min_lr):
        numb_decrements = len(step_size)  # reduce every time we start another cycle
        max_lr_reducer = (max_lr - min_lr) / numb_decrements

        # gradually reduce max_lr, but not below min_lr
        maxlr = max((max_lr-self.cur_decrement*max_lr_reducer), min_lr)
        self.cur_decrement += 1
        return maxlr

# def getTriangularLRs(numb_iterations,max_lr, min_lr, step_size =None):
#     '''
#     returns a single tooth /\ of a sawtooth wave that varies from min_lr to max_lr
#    :param numb_iterations:  total learning_rates to generate, make an even number so stepsize will be 1/2 of it
#     :param max_lr:
#     :param min_lr:
#     :param  step_size =None  used for learning rate finder do several epochs, with increasing LRs
#             to determine appropriate range
#     :return: list of learning rates
#     '''
#     data = []
#     #set stepsize == numb_iterations (1/2 sawtooth) for calculating appropriate Learning rate range
#     if step_size == None:
#         stepsize = numb_iterations//2
#
#     for itr in range(numb_iterations):
#         cycle = math.floor(1+ itr/(2*stepsize))
#         x=abs(itr/stepsize -2*cycle +1)
#         lr = max_lr +(max_lr - min_lr)*max(0,(1-x))
#         data.append(lr)
#     return data
#
#
# def getCosignLRs(numb_iterations,max_lr, min_lr):
#     '''
#     generates a list of max_iter_per_cycle learning rates that starts at max_lr and descends
#     to base_lr in a cosign wave
#     max_iter_per_cycle changes according to schedule in epochs_per_cycle_schedule
#     :param numb_iterations:  total learning_rates to generate, make an even number so stepsize will be 1/2 of it
#     :param max_lr:
#     :param min_lr:
#      :return: list of learning rates
#     '''
#     # cosign varies between 1 and -1 (sweep of 2), factor to scale between lowlr and high_lr
#     scale = 2 / (max_lr - min_lr)
#
#     # then translate so ranges between low_lr and high_lr
#     data = (np.cos(np.linspace(0, np.pi, numb_iterations)) / scale) + (max_lr - min_lr) / 2 + min_lr
#     return data

NUMB_IMAGES = 5011  # numb images in VOC2007
class CyclicLR_Scheduler(object):
    '''
       Part of stochastic gradient descent with warm restarts
       be careful with the learning rate, set it too high and you blow your model

       be sure the number of batches you run is equal to sum(step_size)*2 to get the learning rate to min_lr
       for example, step_size = [5,5,5,5,10,10,10]  then number batches = 50*2=100

       best to use some sort of learning rate finder

    '''
    RESTART_CYCLE = 0
    NUMBER_STEPS_PER_CYCLE = 2
    def __init__(self, optimizer,*,min_lr, max_lr, LR,LR_anneal=None,
                 batch_size = 64, numb_images= NUMB_IMAGES,step_size=[2], writer =None):
        '''
        :param optimizer:  optimizer, used primarily for applying learning rate to params
        :param min_lr:
        :param max_lr:
        :param LR:  calculates a single cycle of the learning rate
        :param LR_anneal:  anneals the learning rate if present
        :param base_lr:  the minimum learning rate
        :param max_lr:  the maximum learning rate
        :param batch_size:
        :param numb_images:
        :param step_size:number of training images per 1/2 cycle, authors suggest setting it between 2 and 10
                        can be a list, if 4 the whole cycle has 2*(4*numb_images//batch_size) = #iterations/cycle
                        this should sum to total number of epochs or your last epoch has lr somewhere
                        between max_lr and base_lr
        :param writer: tensorboard writer (see above)
        usage:
        >>>lr = TriangularLR(max_lr, min_lr)
        >>>lra = LR_anneal_linear(step_size,max_lr,min_lr)
        >>>crs = CyclicLR_Scheduler(optimizer, LR,LR_anneal)

        '''

        #apply learning rates to optimizer layers
        self.optimizer = optimizer

        self.min_lr = min_lr
        self.max_lr=max_lr;

        #learning rate object
        self.LR = LR

        #learning rate annealer
        self.LR_anneal = LR_anneal

        #how many iterations per epoch
        self.batch_size = batch_size
        self.numb_images = numb_images

        #TODO fix this
        # covers the last partial batch
        # self.pad = 1 if numb_images % self.batch_size > 0 else 0
        self.pad=0

        #number iterations per batch
        self.max_iter_per_epoch = numb_images // batch_size + self.pad

        # if [1,2,3] then first cosign descent occurs over 1*2 epoch,
        # second over 2*2 epochs, 3rd over 3*2 epochs for a total of 12 epochs
        #any after that are over 6 epochs
        self.step_size = step_size
        self.step_size_loc =CyclicLR_Scheduler.RESTART_CYCLE  #when incremented starts at 0


        #start at 0
        self.loc = CyclicLR_Scheduler.RESTART_CYCLE

        #the list of lrs periodically refreshed
        self.vals = []
        self.writer = writer

    def _getNumberIterations(self):
        '''
        how many iterations for this cycle
        :return:
        '''
        n_itr = ((self.step_size[self.step_size_loc])*self.max_iter_per_epoch )*CyclicLR_Scheduler.NUMBER_STEPS_PER_CYCLE

        # only advance if there are more, otherwise stay at the last one
        if (self.step_size_loc < len(self.step_size)-1):
            self.step_size_loc += 1  # go to the next one
        return n_itr

    def _calculate_learning_rate_range(self):
        '''
        generates a list of max_iter_per_cycle learning rates that starts at max_lr and descends
        to base_lr in a cosign wave
        max_iter_per_cycle changes according to schedule in epochs_per_cycle_schedule
        '''

        #gradually reduce the learning rate
        maxlr = self.max_lr
        if self.LR_anneal is not None:
            maxlr = self.LR_anneal.getMaxLR(step_size=self.step_size,max_lr= self.max_lr,min_lr=self.min_lr)

        # then translate so ranges between low_lr and high_lr
        self.vals = self.LR.getLRs(numb_iterations = self._getNumberIterations(), max_lr=maxlr, min_lr=self.min_lr)

    def _get_lr(self):
        #time to restart?
        if (self.loc == len(self.vals)):
            self.loc = CyclicLR_Scheduler.RESTART_CYCLE

        # time to go to the next cycle length?
        if (self.loc == CyclicLR_Scheduler.RESTART_CYCLE):
            self._calculate_learning_rate_range()

        lrs=[]
        lrs.append(self.vals[self.loc])

        #prepare for the next one
        self.loc += 1

        if (self.writer is not None):
            self.writer('CLRS', lrs[0])
        # print(f"     lr{lrs[0]}")
        return lrs

    def batch_step(self):
         for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr
         self.cur_lr = lr   #used in learning rate finder


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



