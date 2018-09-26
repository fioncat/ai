import random

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            self.t += 1
        else:
            self.t += 1
            self.epsilon -= 0.1

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        self.Qtable.setdefault(state, {a: 0.0 for a in self.valid_actions})

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():
            return random.random() < self.epsilon

        if self.learning:
            if is_random_exploration():
                return_random = True
            else:
                return_random = False

        elif self.testing:
            return_random = False
        else:
            return_random = True
        
        if return_random:
            return random.choice(self.valid_actions)
        else:
            # max_val = max([v for _, v in self.Qtable[self.state].items()])
            # return [k for k, v in self.Qtable[self.state].items() if v == max_val][0]
            return max(self.Qtable[self.state], key=self.Qtable[self.state].get)

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """

        def get_max_q(state):
            return max(self.Qtable[next_state].values())

        if self.learning:
            self.Qtable[self.state][action] = (1 - self.alpha) * self.Qtable[self.state][action] +\
                                                self.alpha * (r + self.gamma * get_max_q(next_state))

    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        return action, reward

if __name__ == '__main__':
    epoch = 30

    epsilon0 = 0.9 
    alpha = 0.7
    gamma = 0.9

    maze_size = (12, 12)
    trap_number = 4
    from Runner import Runner
    from Maze import Maze

    g = Maze(maze_size=maze_size,trap_number=trap_number)
    r = Robot(g,alpha=alpha, epsilon0=epsilon0, gamma=gamma)
    r.set_status(learning=True)

    runner = Runner(r, g)
    runner.run_training(epoch, display_direction=True)
    runner.generate_movie(filename = "final3.mp4")
