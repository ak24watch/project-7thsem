import os
import flax.serialization
import flax.nnx as nnx
from training_dir.train_cno import build_cno_model

def load_cno_model(config: dict, load_path: str):
    """
    Load a saved Flax NNX CNO_2D model.

    Args:
        config: Same architecture config used for training.
        load_path: Path to the .msgpack checkpoint (e.g., config["save_path"]).

    Returns:
        The restored model.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    rngs = nnx.Rngs(config.get("random_seed", 0))
    model = build_cno_model(config, rngs)

    with open(load_path, "rb") as f:
        payload = f.read()

    state = nnx.state(model)
    loaded_state = flax.serialization.from_bytes(state, payload)
    nnx.update(model, loaded_state)
    return model


if __name__ == "__main__":
    # Example:
    # from training_dir.train_cno import jnp  # if you need to rebuild the same config
    # config = {...}  # same dict you used for training
    # model = load_cno_model(config, config["save_path"])
    pass
