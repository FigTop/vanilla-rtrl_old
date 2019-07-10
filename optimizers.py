'''
Borrowed from
https://gist.github.com/Harhro94/3b809c5ae778485a9ea9d253c4bfc90a
'''

import numpy as np

class Optimizer:

    def __init__(self, allowed_kwargs_, **kwargs):

        allowed_kwargs = {'clipnorm', 'clipvalue', 'lr_decay_rate', 'min_lr'}.union(allowed_kwargs_)
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)

    def clip_norm(self, g, norm):

        if norm>self.clipnorm:
            return (g/norm)*self.clipnorm
        else:
            return g

    def lr_decay(self):

        self.lr_ = self.lr_*self.lr_decay_rate
        try:
            return np.max([self.lr_, self.min_lr])
        except AttributeError:
            return self.lr_


class Adam(Optimizer):

    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):

        allowed_kwargs_ = set()
        super().__init__(allowed_kwargs_, **kwargs)

        self.iterations = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_update(self, params, grads):
        """ params and grads are list of numpy arrays
        """
        original_shapes = [x.shape for x in params]
        params = [x.flatten() for x in params]
        grads = [x.flatten() for x in grads]

        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = np.sqrt(sum([np.sum(np.square(g)) for g in grads]))
            grads = [self.clip_norm(g, norm) for g in grads]

        '''
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        '''

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (np.sqrt(1. - np.power(self.beta_2, t)) /
                     (1. - np.power(self.beta_1, t)))

        if not hasattr(self, 'ms'):
            self.ms = [np.zeros(p.shape) for p in params]
            self.vs = [np.zeros(p.shape) for p in params]

        ret = [None] * len(params)
        for i, p, g, m, v in zip(range(len(params)), params, grads, self.ms, self.vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * np.square(g)
            p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
            self.ms[i] = m_t
            self.vs[i] = v_t
            ret[i] = p_t

        self.iterations += 1

        for i in range(len(ret)):
            ret[i] = ret[i].reshape(original_shapes[i])

        return ret

class SGD(Optimizer):
    """SGD optimizer.
    # Arguments
        lr: float >= 0. Learning rate.
    """

    def __init__(self, lr=0.001, **kwargs):

        allowed_kwargs_ = set()
        super().__init__(allowed_kwargs_, **kwargs)

        self.lr_ = np.copy(lr)
        self.lr = lr

    def get_update(self, params, grads):
        """ params and grads are list of numpy arrays
        """
        original_shapes = [x.shape for x in params]
        params = [x.flatten() for x in params]
        grads = [x.flatten() for x in grads]

        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = np.sqrt(sum([np.sum(np.square(g)) for g in grads]))
            grads = [self.clip_norm(g, norm) for g in grads]

        if hasattr(self, 'lr_decay_rate'):
            self.lr = self.lr_decay()

        """ #TODO: implement clipping
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = np.sqrt(sum([np.sum(np.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        """

        ret = [None] * len(params)
        for i, p, g in zip(range(len(params)), params, grads):
            ret[i] = p - self.lr * g

        for i in range(len(ret)):
            ret[i] = ret[i].reshape(original_shapes[i])

        return ret

class rprop(Optimizer):

    def __init__(self, step_gain=1.2, step_attenuate=0.5, init_step_size=0.1, **kwargs):

        allowed_kwargs_ = {'step_size_lb', 'step_size_ub'}
        super().__init__(allowed_kwargs_, **kwargs)

        self.step_gain = step_gain
        self.step_attenuate = step_attenuate

        self.step_size = init_step_size

    def get_update(self, params, grads):

        pass


