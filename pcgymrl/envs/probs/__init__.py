from pcgymrl.envs.probs.binary_prob import BinaryProblem
from pcgymrl.envs.probs.ddave_prob import DDaveProblem
from pcgymrl.envs.probs.mdungeon_prob import MDungeonProblem
from pcgymrl.envs.probs.sokoban_prob import SokobanProblem
from pcgymrl.envs.probs.zelda_prob import ZeldaProblem
from pcgymrl.envs.probs.smb_prob import SMBProblem

# all the problems should be defined here with its corresponding class
PROBLEMS = {
    "binary": BinaryProblem,
    "ddave": DDaveProblem,
    "mdungeon": MDungeonProblem,
    "sokoban": SokobanProblem,
    "zelda": ZeldaProblem,
    "smb": SMBProblem,
}
