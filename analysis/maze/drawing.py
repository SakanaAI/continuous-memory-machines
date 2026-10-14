"""Image helpers for the paper maze attention plots."""

import numpy as np

_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _draw_predicted_path(maze_rgb, actions, rainbow_cmap):
    """Overlay the predicted action sequence on the maze as a rainbow from start to goal.

    Agent (start) is red, goal is green; intermediate visited cells are blended with
    a colour sampled from `rainbow_cmap` at that step's fraction of the path.
    Stops as soon as the path predicts Wait, steps into a wall, or reaches the goal.
    """
    img = np.copy(maze_rgb)
    starts = np.argwhere(
        (img[..., 0] > 0.8) & (img[..., 1] < 0.2) & (img[..., 2] < 0.2)
    )  # red = agent / start
    if len(starts) == 0:
        return img
    r, c = starts[0]
    H, W = img.shape[:2]
    colours = rainbow_cmap(np.linspace(0.0, 1.0, max(len(actions), 1)))[:, :3]
    for step_idx, action in enumerate(actions):
        if action == 4:  # Wait → path finished
            break
        if action not in _DELTAS:
            break
        dr, dc = _DELTAS[int(action)]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            break
        # Stop at wall (black)
        if img[nr, nc].sum() < 0.2:
            break
        is_goal = img[nr, nc, 1] > 0.8 and img[nr, nc, 0] < 0.2 and img[nr, nc, 2] < 0.2
        is_start = (
            img[nr, nc, 0] > 0.8 and img[nr, nc, 1] < 0.2 and img[nr, nc, 2] < 0.2
        )
        if not (is_goal or is_start):
            img[nr, nc] = 0.4 * img[nr, nc] + 0.6 * colours[step_idx]
        r, c = nr, nc
        if is_goal:  # reached green goal
            break
    return img


def _prepare_maze_image(inputs):
    """Take a (1, 3, H, W) tensor in [-1, 1] and return an (H, W, 3) array in [0, 1]."""
    img = inputs[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = (img + 1) / 2  # [-1, 1] → [0, 1]
    return np.clip(img, 0, 1)
