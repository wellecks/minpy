import time
import pickle
import os

import minpy.numpy as np
from minpy import core
from minpy.nn.solver import Solver

class RLPolicyGradientSolver(Solver):
    """A custom `Solver` for policy gradient models.
    Specifically, the model should provide:
        .forward(X)
        .choose_action(p)
        .loss(xs, ys, rs)
        .discount_rewards(rs)
        .preprocessor
    """
    def __init__(self, model, env, **kwargs):
        self.model = model
        self.env = env
        self.update_every = kwargs.pop('update_every', 10)
        self.num_episodes = kwargs.pop('num_episodes', 100000)
        self.save_every = kwargs.pop('save_every', 10)
        self.save_dir = kwargs.pop('save_dir', '')
        self.resume_from = kwargs.pop('resume_from', None)
        self.render = kwargs.pop('render', False)

        self.running_reward = None
        self.episode_reward = 0

        super(RLPolicyGradientSolver, self).__init__(model, None, None, **kwargs)

    def _reset_data_iterators(self):
        # This `Solver` does not use data iterators.
        pass

    def init(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.resume_from is not None:
            with open(self.resume_from) as f:
                params = pickle.load(f)
            self.model.params = {k: np.array(v.tolist()) for k, v in params.iteritems()}
        else:
            super(RLPolicyGradientSolver, self).init()

    def run_episode(self):
        """Run an episode (multiple games) using the current model to generate training data.
        :return tuple (xs, ys, rs): The N x input_size observations, N x 1 action labels,
                and N x 1 discounted rewards obtained from running the episode's N steps.
        """
        observation = self.env.reset()
        self.model.preprocessor.reset()
        self.episode_reward = 0

        xs, ys, rs = [], [], []
        done = False
        game_number = 1
        game_start = time.time()
        while not done:
            if self.render:
                self.env.render()
            x = self.model.preprocessor.preprocess(observation)
            p = self.model.forward(x)
            a, y = self.model.choose_action(p.asnumpy().ravel()[0])
            observation, r, done, info = self.env.step(a)

            xs.append(x.asnumpy().ravel())
            ys.append(y)
            rs.append(r)
            self.episode_reward += r
            if self._game_complete(r):
                game_time = time.time() - game_start
                if self.verbose:
                    print('game %d complete (%.2fs), reward: %f' % (game_number, game_time, r))
                game_number += 1
                game_start = time.time()

        # Episode finished.
        self.running_reward = self.episode_reward if self.running_reward is None else (
            0.99*self.running_reward + 0.01*self.episode_reward)
        xs = np.vstack(xs)
        ys = np.vstack(ys)
        rs = np.expand_dims(self.model.discount_rewards(rs), axis=1)
        return xs, ys, rs

    def _game_complete(self, reward):
        return reward != 0

    def train(self):
        grad_buffer = self._init_grad_buffer()

        for episode_number in xrange(1, self.num_episodes):
            episode_start = time.time()
            # Generate an episode of training data
            xs, ys, rs = self.run_episode()

            # Compute loss and gradient
            def loss_func(*params):
                ps = self.model.forward(xs)
                return self.model.loss(ps, ys, rs)

            param_arrays = list(self.model.params.values())
            param_keys = list(self.model.params.keys())
            grad_and_loss_func = core.grad_and_loss(loss_func, argnum=range(len(param_arrays)))
            backward_start = time.time()
            grad_arrays, loss = grad_and_loss_func(*param_arrays)
            backward_time = time.time() - backward_start
            grads = dict(zip(param_keys, grad_arrays))

            # Accumulate gradients until an update is performed.
            for k, v in grads.iteritems():
                grad_buffer[k] += v

            self.loss_history.append(loss.asnumpy())
            episode_time = time.time() - episode_start

            if self.verbose:
                print('Backward pass complete (%.2fs)' % backward_time)
            if self.verbose or episode_number % self.print_every == 0:
                print('Episode %d complete (%.2fs), loss: %s, reward: %s, running reward: %s' %
                      (episode_number, episode_time, loss, self.episode_reward, self.running_reward))

            if episode_number % self.update_every == 0:
                # Perform parameter update.
                for p, w in self.model.params.items():
                    dw = grad_buffer[p]
                    config = self.optim_configs[p]
                    next_w, next_config = self.update_rule(w, dw, config)
                    self.model.params[p] = next_w
                    self.optim_configs[p] = next_config
                    grad_buffer[p] = np.zeros_like(w)

            if episode_number % self.save_every == 0:
                # Save model parameters.
                if self.verbose:
                    print('Saving model parameters...')
                file_name = os.path.join(self.save_dir, 'params_%d.p' % episode_number)
                with open(file_name, 'w') as f:
                    pickle.dump({k: v.asnumpy() for k, v in self.model.params.iteritems()}, f)
                if self.verbose:
                    print('Wrote parameter file %s' % file_name)

    def _init_grad_buffer(self):
        return {k: np.zeros_like(v) for k, v in self.model.params.iteritems()}
