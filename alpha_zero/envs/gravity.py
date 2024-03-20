
"""Gravity env class."""
from .gomoku import *

class GravityEnv(GomokuEnv):
    """Gravity env class."""
    def __init__(self, board_size: int = 15, num_to_win: int = 4, num_stack: int = 8) -> None:
        super().__init__(board_size, num_to_win, num_stack)
        self.id = "GravityYonmoku"

        # Ground actions are only legal
        self.legal_actions[self.board_size:] = 0


    def reset(self, **kwargs) -> np.ndarray:
        """Resets the game to initial state."""
        super().reset(**kwargs)
        
        # Ground actions are only legal
        self.legal_actions[self.board_size:] = 0


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Plays one move."""
        if self.is_game_over():
            raise RuntimeError('Game is over, call reset before using step method.')
        if action is not None and action != self.resign_move and not 0 <= int(action) <= self.action_space.n - 1:
            raise ValueError(f'Invalid action. The action {action} is out of bound.')
        if action is not None and action != self.resign_move and self.legal_actions[int(action)] != 1:
            raise ValueError(f'Illegal action {action}.')

        self.last_move = copy(int(action))
        self.last_player = copy(self.to_play)
        self.steps += 1

        self.add_to_history(self.last_player, self.last_move)

        reward = 0.0

        # Make sure the action is illegal from now on.
        self.legal_actions[action] = 0
        # Make sure the upper action is legal from now on.
        if action // self.board_size < self.board_size - 1:
            self.legal_actions[action + self.board_size] = 1

        # Update board state.
        row_index, col_index = self.action_to_coords(action)
        self.board[row_index, col_index] = self.to_play

        # Make sure the latest board position is always at index 0
        self.board_deltas.appendleft(np.copy(self.board))

        # The reward is always computed from last player's perspective
        # which follows standard MDP practice, where reward function r_t = R(s_t, a_t)
        if self.is_current_player_won():
            reward = 1.0
            self.winner = self.to_play

        done = self.is_game_over()

        # Switch next player
        self.to_play = self.opponent_player

        return self.observation(), reward, done, {}