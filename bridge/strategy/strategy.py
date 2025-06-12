"""High-level strategy code"""

# !v DEBUG ONLY
from time import time

from bridge import const
from bridge.auxiliary import aux, fld, rbt
from bridge.const import State as GameStates
from bridge.router.base_actions import Action, Actions, KickActions


class Strategy:
    """Main class of strategy"""

    def __init__(
            self,
    ) -> None:
        self.we_active = False

    def process(self, field: fld.Field) -> list[Action]:
        """Game State Management"""
        if field.game_state not in [GameStates.KICKOFF, GameStates.PENALTY]:
            if field.active_team in [const.Color.ALL, field.ally_color]:
                self.we_active = True
            else:
                self.we_active = False

        actions: list[Action] = []
        for _ in range(const.TEAM_ROBOTS_MAX_COUNT):
            actions.append(Actions.Stop())

        if field.ally_color == const.COLOR:
            text = str(field.game_state) + "  we_active:" + str(self.we_active)
            field.strategy_image.print(aux.Point(600, 780), text, need_to_scale=False)
        match field.game_state:
            case GameStates.RUN:
                self.run(field, actions)
            case GameStates.TIMEOUT:
                pass
            case GameStates.HALT:
                return actions
            case GameStates.PREPARE_PENALTY:
                pass
            case GameStates.PENALTY:
                pass
            case GameStates.PREPARE_KICKOFF:
                pass
            case GameStates.KICKOFF:
                pass
            case GameStates.FREE_KICK:
                pass
            case GameStates.STOP:
                pass

        return actions

    def run(self, field: fld.Field, actions: list[Action]) -> None:
            """
            Assigning roles to robots and managing them
                roles - robot roles sorted by priority
                robot_roles - list of robot id and role matches
            """
            idx = 0
            pos = aux.Point(-1000, -1000)
            actions[idx] = Actions.GoToPointIgnore(pos, 1.0)



        
