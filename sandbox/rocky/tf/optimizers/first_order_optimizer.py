


from rllab.misc import ext
from rllab.misc import logger
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.misc import tensor_utils
# from rllab.algo.first_order_method import parse_update_method
from rllab.optimizers.minibatch_dataset import BatchDataset
from collections import OrderedDict
import tensorflow as tf
import time
from functools import partial
import tqdm


class FirstOrderOptimizer(Serializable):
    """
    Performs (stochastic) gradient descent, possibly using fancier methods like adam etc.
    """

    def __init__(
            self,
            tf_optimizer_cls=None,
            tf_optimizer_args=None,
            # learning_rate=1e-3,
            max_epochs=1000,
            tolerance=1e-6,
            batch_size=32,
            callback=None,
            verbose=False,
            **kwargs):
        """

        :param max_epochs:
        :param tolerance:
        :param update_method:
        :param batch_size: None or an integer. If None the whole dataset will be used.
        :param callback:
        :param kwargs:
        :return:
        """
        Serializable.quick_init(self, locals())
        self._opt_fun = None
        self._target = None
        self._callback = callback
        if tf_optimizer_cls is None:
            tf_optimizer_cls = tf.train.AdamOptimizer
        if tf_optimizer_args is None:
            tf_optimizer_args = dict(learning_rate=1e-3)
        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._verbose = verbose
        self._input_vars = None
        self._train_op = None
        # hack to figure out whether optimize() has been called before (ugh,
        # need something better)
        self._num_opt_calls = 0

    def update_opt(self, loss, target, inputs, extra_inputs=None, summary_writer=None, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        # I replaced the next line with the ones which follow so that I could
        # give gradients to Tensorboard
        # self._train_op = self._tf_optimizer.minimize(loss, var_list=target.get_params(trainable=True))
        params = target.get_params(trainable=True)
        grads_and_vars = self._tf_optimizer.compute_gradients(
            loss, var_list=params)
        self._train_op = self._tf_optimizer.apply_gradients(
            grads_and_vars=grads_and_vars)
        for g, v in grads_and_vars:
            tf.summary.histogram(
                'weight_grads/' + v.name, g, collections=['dist_info_sym'])
            for slot in self._tf_optimizer.get_slot_names():
                slot_var = self._tf_optimizer.get_slot(v, slot)
                if slot_var is not None:
                    dest_name = 'slots-' + slot + '/' + v.name
                    tf.summary.histogram(
                        dest_name, slot_var, collections=['dist_info_sym'])

        # TODO: need to get a better binding between name and network here
        # add weights as well
        weight_op = tf.summary.merge_all('weights')
        tf.summary.merge([weight_op], collections=['dist_info_sym'])
        self._summary_op = tf.summary.merge_all('dist_info_sym')
        self._summary_writer = summary_writer

        # updates = OrderedDict([(k, v.astype(k.dtype)) for k, v in updates.iteritems()])

        if extra_inputs is None:
            extra_inputs = list()
        self._input_vars = inputs + extra_inputs
        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs + extra_inputs, loss),
        )

    def loss(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_loss"](*(tuple(inputs) + extra_inputs))

    def optimize(self, inputs, extra_inputs=None, callback=None):

        if len(inputs) == 0:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError

        f_loss = self._opt_fun["f_loss"]

        if extra_inputs is None:
            extra_inputs = tuple()

        last_loss = f_loss(*(tuple(inputs) + extra_inputs))

        start_time = time.time()

        dataset = BatchDataset(inputs, self._batch_size, extra_inputs=extra_inputs)

        sess = tf.get_default_session()

        for epoch in range(self._max_epochs):
            batch_iter = enumerate(dataset.iterate(update=True))
            if self._verbose:
                logger.log("Epoch %d" % (epoch))
                batch_iter = tqdm(batch_iter)

            for batch_idx, batch in batch_iter:
                feed_dict = dict(list(zip(self._input_vars, batch)))
                if self._summary_op is not None:
                    do_summary = (batch_idx % 10) == 0
                    if do_summary:
                        _, summary = sess.run([self._train_op, self._summary_op], feed_dict)
                        self._summary_writer.add_summary(summary)
                else:
                    sess.run(self._train_op, feed_dict)

            new_loss = f_loss(*(tuple(inputs) + extra_inputs))

            if self._verbose:
                logger.log("Epoch: %d | Loss: %f" % (epoch, new_loss))
            if self._callback or callback:
                elapsed = time.time() - start_time
                callback_args = dict(
                    loss=new_loss,
                    params=self._target.get_param_values(trainable=True) if self._target else None,
                    itr=epoch,
                    elapsed=elapsed,
                )
                if self._callback:
                    self._callback(callback_args)
                if callback:
                    callback(**callback_args)

            if abs(last_loss - new_loss) < self._tolerance:
                break
            last_loss = new_loss

        self._num_opt_calls += 1
