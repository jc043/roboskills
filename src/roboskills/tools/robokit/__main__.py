import tyro

from roboskills.tools.robokit.humanoid import retarget_person
from roboskills.tools.robokit.mpc import track_with_mpc
from roboskills.tools.robokit.retarget import retarget_hand


tyro.extras.subcommand_cli_from_dict({"retarget": retarget_hand, "mpc": track_with_mpc, "humanoid": retarget_person})
