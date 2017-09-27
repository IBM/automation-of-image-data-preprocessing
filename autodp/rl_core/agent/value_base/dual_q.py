"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from autodp.rl_core.agent.base_agent import BaseAgent
from autodp.config.cf_container import Config as cf
from autodp.utils.misc import get_class


class DualQ(BaseAgent):
    """
    This class implements a RL algorithm using a dual-q network.
    """
    def __init__(self, cont):
        """
        Initialization.
        """
        super().__init__(cont)

    def setup_policy(self):
        """
        Build network to approximate an action-value function.
        :return:
        """
        # Construct a reinforcement learning graph
        rl_graph_class = get_class(cf.rl_graph)
        rl_graph = rl_graph_class(net_arch=cf.rl_arch, loss_func=cf.rl_loss,
                                  name="rl_graph")

        return rl_graph

    def train_policy(self):
        """
        Policy improvement and evaluation.
        :return:
        """


        step_drop = (cf.exploration - cf.min_explore) / cf.anneal_step

        # Start training the policy network
        early_stop, best_valid = 0, sys.maxsize
        for epoch in range(cf.num_epoch):
            # Initialize an environment
            env = BatchSimEnv()
            image_batch, qouts, batch_size = [], [], cf.batch_size

            # Process an epoch
            num_step, reward_all, done_all = 0, 0, 0
            with bz2.BZ2File(cf.train_path, "rb") as df:
                while True:
                    # Add more images for batch processing
                    if batch_size > 0:
                        images, labels = self._get_batch(df, batch_size)
                        image_batch.extend(images)
                        qouts.extend([0] * len(images))
                        env.add(image_batch=images, label_batch=labels)

                    # If no image left, then exit the current epoch
                    if len(image_batch) == 0:
                        print("Epoch %d, final step has accumulated "
                              "rewards %g and processed %d images"
                              % (epoch, reward_all, done_all))
                        break

                    # Do exploration
                    if np.random.rand(1) < cf.exploration:
                        # if np.random.rand(1) < 1.1:
                        actions = np.random.randint(0, cf.num_action,
                                                    len(image_batch))
                        # else:
                        #    actions = np.random.randint(0, 2,
                        #                                len(image_batch))

                    # Select actions using the policy network
                    else:
                        [actions, qout] = self._sess.run(
                            [rg.get_next_actions, rg.get_qout],
                            feed_dict={rg.get_instances: image_batch})
                        qouts = list(np.array(qouts) + qout[:, 0] - qout[:, 1])

                    # Delay adding extra examples to the buffer
                    # after doing actions
                    env.step(actions, qouts)
                    extra = self._add_extra_example(env)
                    exp_buf.add(extra)

                    # Decrease the exploration
                    if cf.exploration > cf.min_explore:
                        cf.exploration -= step_drop

                    # Train and update the policy network
                    if num_step % cf.update_freq == 0 and exp_buf.get_size() > cf.train_size:
                        train_batch = exp_buf.sample(cf.train_size)
                        # train_batch.extend(extra)
                        # train_batch = extra
                        i_states = np.array([e[0] for e in train_batch])
                        i_actions = np.array([e[1] for e in train_batch])
                        o_states = np.array([e[2] for e in train_batch])
                        i_rewards = np.array([e[3] for e in train_batch])
                        end_mul = np.array([1 - e[4] for e in train_batch])

                        qmax = self._sess.run(rg.get_qmax,
                                              feed_dict={rg.get_instances: o_states})
                        target = i_rewards + cf.gamma * qmax * end_mul
                        target[np.where(target < -1)] = -1
                        target[np.where(target > 1)] = 1
                        #    cf.gamma-0.02*i_ages)*qmax*end_mul

                        # for (i, q) in enumerate(qmax):
                        #    if q < 0: target[i] = -1

                        [_, err] = self._sess.run(
                            [rg.get_train_step, rg.get_error],
                            feed_dict={rg.get_instances: i_states,
                                       rg.get_actions: i_actions,
                                       rg.get_targets: target})

                    # Add extra examples to the buffer
                    # exp_buf.add(extra)

                    # Update input data after 1 step
                    rewards, dones, states, _, _ = self._compute_done(env)
                    reward_all += sum(rewards)
                    done_all += sum(dones)
                    num_step += 1
                    batch_size = sum(dones)
                    image_batch = list(compress(states,
                                                np.logical_not(dones)))
                    qouts = list(compress(qouts, np.logical_not(dones)))
                    env.update_done(dones)

                    # Print rewards after every number of steps
                    if num_step % 100 == 0:
                        print("Epoch %d, step %d has accumulated "
                              "rewards %g and processed %d images "
                              "and train error %g" % (epoch, num_step,
                                                      reward_all, done_all, err))

            re_err, valid_err, _, _ = self._valid_test(cf.valid_path, rg)
            if re_err < best_valid:
                best_valid = re_err
                early_stop = 0
                print("Best valid reward is %g" % (-best_valid))

                # Save model
                clear_model_dir(os.path.dirname(cf.save_model))
                saver = tf.train.Saver(tf.global_variables())
                saver.save(self._sess, cf.save_model)

                # Save exp
                to_pickle(os.path.dirname(cf.save_model) + "/exp.pkl",
                          exp_buf)

            else:
                early_stop += 1

            if early_stop >= 3000:
                if epoch > 3:
                    print("Exit due to early stopping")
                    break

                else:
                    early_stop = 0





