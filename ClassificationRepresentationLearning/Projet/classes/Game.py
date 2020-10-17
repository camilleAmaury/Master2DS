import numpy as np

class Game(object):
    
    def __init__(self, time, neigtbors):
        super(Game, self).__init__()
        self._const_time = time
        self.neigtbors = neigtbors
    
    # init a game and reset it
    def init_game(self, M, init_coord):
        self._Values = M[:,1:]
        self._Ids = M[:, 0].astype(int)
        self.actual_coord = (init_coord[0], init_coord[1], -1)
        self.ids = []
        self.rewards = []
        self.time = 0.
        # first step
        possibilities = self.neigtbors[self.actual_coord[2]]
        rewards = [self._Values[int(possibilities[i][0]), 2] for i in range(len(possibilities))]
        return False, False, possibilities, rewards, self._const_time - self.time, self.actual_coord
        
    # step in a game
    def step_env(self, action, distance):
        # check if there is a correct action or not
        if action == -1:
            return True, False, None, None, self._const_time - self.time, None
        else:
            # get reward
            reward = self._Values[action, 2]
            self._Values[action, 2] = 0.

            # keep track of action
            self.ids.append(self._Ids[action])
            self.rewards.append(reward)

            # update time and coordinate
            self.time += distance
            self.actual_coord = (self._Values[action, 0], self._Values[action, 1], action)

            # step into the environnement
            if self.time < self._const_time:
                possibilities = self.neigtbors[self.actual_coord[2]]
                rewards = [self._Values[int(possibilities[i][0]), 2] for i in range(len(possibilities))]
                return False, False, possibilities, rewards, self._const_time - self.time, self.actual_coord
            else:
                # previous time was not correctly endled by the agent(strategy)
                return False, True, None, None, self._const_time - self.time, None
    
        
    
    # string representation of the object
    def __repr__(self):
        return "<Object : Game (DataCamp) <> time:{}>".format(self._const_time)
    def __str__(self):
        return self.__repr__()

class Game2(object):
    
    def __init__(self, time, neigtbors, m, n):
        super(Game2, self).__init__()
        self._const_time = time
        self.neigtbors = neigtbors
        self.n = n
        self.m = m
    
    # init a game and reset it
    def init_game(self, M, init_coord):
        self._Values = M[:,1:]
        self._Ids = M[:, 0].astype(int)
        self.actual_coord = (init_coord[0], init_coord[1], -1)
        self.ids = []
        self.rewards = []
        self.time = 0.
        # first step
        possibilities = self.neigtbors[self.actual_coord[2]].tolist()
        rewards = []
        new_possibilities = []
        counter = 0
        for i in range(len(possibilities)):
            if counter == self.m:
                break
            id_ = int(possibilities[i][0])
            reward = self._Values[id_, 2]
            if reward != 0.:
                children = self.neigtbors[id_]
                c_rewards = [self._Values[int(children[j][0]), 2] for j in range(self.n)]
                rewards.append(
                    (
                        reward,
                        sum([c_rewards[j] / children[j][1]**3  if children[j][1] != 0. and c_rewards[j] != 0. else 0. for j in range(self.n)])/self.n
                    )
                )
                new_possibilities.append(possibilities[i])
                counter+=1
        return False, False, new_possibilities, rewards, self._const_time - self.time, self.actual_coord
        
    # step in a game
    def step_env(self, action, distance):
        # check if there is a correct action or not
        if action == -1:
            return True, False, None, None, self._const_time - self.time, None
        else:
            # get reward
            reward = self._Values[action, 2]
            self._Values[action, 2] = 0.

            # keep track of action
            self.ids.append(self._Ids[action])
            self.rewards.append(reward)

            # update time and coordinate
            self.time += distance
            self.actual_coord = (self._Values[action, 0], self._Values[action, 1], action)

            # step into the environnement
            if self.time < self._const_time:
                possibilities = self.neigtbors[self.actual_coord[2]].tolist()
                rewards = []
                counter = 0
                new_possibilities = []
                for i in range(len(possibilities)):
                    if counter == self.m:
                        break
                    id_ = int(possibilities[i][0])
                    reward = self._Values[id_, 2]
                    if reward != 0.:
                        children = self.neigtbors[id_]
                        c_rewards = [self._Values[int(children[j][0]), 2] for j in range(self.n)]
                        rewards.append(
                            (
                                reward,
                                sum([c_rewards[j] / children[j][1]**3  if children[j][1] != 0. and c_rewards[j] != 0. else 0. for j in range(self.n)])/self.n
                            )
                        )
                        new_possibilities.append(possibilities[i])
                        counter+=1
                return False, False, new_possibilities, rewards, self._const_time - self.time, self.actual_coord
            else:
                # previous time was not correctly endled by the agent(strategy)
                return False, True, None, None, self._const_time - self.time, None
    
        
    
    # string representation of the object
    def __repr__(self):
        return "<Object : Game (DataCamp) <> time:{}>".format(self._const_time)
    def __str__(self):
        return self.__repr__()


def play_game(game_, M, strategy, init_coord, epochs=1):
    game = game_
    
    for _ in range(epochs):
        # reset env
        game_correctly_ended, game_bug_ended, possibilities, rewards, time_left, coordinates = game.init_game(M, init_coord)
        _action = None
        # stepping in the environnement
        while((not game_correctly_ended) and (not game_bug_ended)):
            # the agent choose the following action
            action, distance = strategy.choose_action(possibilities, rewards, time_left, coordinates)
            if _action != None and _action == action:
                print("bug")
            # step
            game_correctly_ended, game_bug_ended, possibilities, rewards, time_left, coordinates = game.step_env(action, distance)
            _action=action
        #print("Time consumed :\n   > {}".format(game.time))
    return game.ids, sum(game.rewards)