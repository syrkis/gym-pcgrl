from pcgym.envs.probs.binary_prob import BinaryProblem
from pcgym.envs.probs.ddave_prob import DDaveProblem
from pcgym.envs.probs.mdungeon_prob import MDungeonProblem
from pcgym.envs.probs.sokoban_prob import SokobanProblem
from pcgym.envs.probs.zelda_prob import ZeldaProblem
from pcgym.envs.probs.smb_prob import SMBProblem

# all the problems should be defined here with its corresponding class
PROBLEMS = {
    "binary": BinaryProblem,
    "ddave": DDaveProblem,
    "mdungeon": MDungeonProblem,
    "sokoban": SokobanProblem,
    "zelda": ZeldaProblem,
    "smb": SMBProblem,
}
