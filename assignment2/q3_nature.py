import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: 
            - You may find the following functions useful:
                - tf.layers.conv2d
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################

        # image height - 84 x 84 x 4 - 4th channel is the alpha channel
        conv1 = layers.conv2d(inputs=state, kernel_size=(8, 8), num_outputs=32, stride=4, scope=scope, reuse=reuse, name='conv1')
        relu1 = tf.nn.relu(conv1, name='relu1')

        conv2 = layers.conv2d(inputs=relu1, kernel_size=(4, 4), num_outputs=64, stride=2, scope=scope, reuse=reuse, name='conv2')
        relu2 = tf.nn.relu(conv2, name='relu2')

        conv3 = layers.conv2d(inputs=relu2, kernel_size=(3, 3), num_outputs=64, stride=1, scope=scope, reuse=reuse,
                              name='conv3')
        relu3 = tf.nn.relu(conv3, name='relu3')

        dense1 = layers.dense(inputs=relu3, units=512, scope=scope, reuse=reuse, name='dense1')
        relu4 = tf.nn.relu(dense1, name='dense1')

        dense2 = layers.dense(inputs=relu4, units=num_actions, scope=scope, reuse=reuse, name='dense2')
        out = layers.flatten(dense2, name='flatten1')

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
