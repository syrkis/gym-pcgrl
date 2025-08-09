from pcgym.envs.reps.narrow_rep import NarrowRepresentation
from pcgym.envs.reps.narrow_cast_rep import NarrowCastRepresentation
from pcgym.envs.reps.narrow_multi_rep import NarrowMultiRepresentation
from pcgym.envs.reps.wide_rep import WideRepresentation
from pcgym.envs.reps.turtle_rep import TurtleRepresentation
from pcgym.envs.reps.turtle_cast_rep import TurtleCastRepresentation

# all the representations should be defined here with its corresponding class
REPRESENTATIONS = {
    "narrow": NarrowRepresentation,
    "narrowcast": NarrowCastRepresentation,
    "narrowmulti": NarrowMultiRepresentation,
    "wide": WideRepresentation,
    "turtle": TurtleRepresentation,
    "turtlecast": TurtleCastRepresentation,
}
