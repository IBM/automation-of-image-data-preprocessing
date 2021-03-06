from autodp.rl_core.env.action.base_action import BaseAction
from autodp import cf


@BaseAction.register
class SimpleAction(BaseAction):
    """This class implements simple actions on an image such as flip, rotate, etc."""
    def apply_action(self, action, cur_state, label):
        """Main method to apply an action within an environment."""
        if action < cf.num_class:
            state = cur_state
            if label == action:
                reward = cf.num_class - 1
            else:
                reward = -1
            done = True
        elif action == cf.num_class + 0:
            state = self._flip(cur_state, -1)
            reward = 0
            done = False
        elif action == cf.num_class + 1:
            state = self._flip(cur_state, 0)
            reward = 0
            done = False
        elif action == cf.num_class + 2:
            state = self._flip(cur_state, 1)
            reward = 0
            done = False
        elif action == cf.num_class + 3:
            state = self._rotate(cur_state, 1)
            reward = 0
            done = False
        elif action == cf.num_class + 4:
            state = self._rotate(cur_state, 2)
            reward = 0
            done = False
        elif action == cf.num_class + 5:
            state = self._rotate(cur_state, -1)
            reward = 0
            done = False
        elif action == cf.num_class + 6:
            state = self._rotate(cur_state, -2)
            reward = 0
            done = False
        return state, reward, done
