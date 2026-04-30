"""Efficient global-ish jigsaw reconstruction for rotated map patches.

The competition data is a square jigsaw puzzle: every patch is used once,
rotations are allowed, and patch_0 is the fixed top-left anchor.  A greedy
left-to-right fill is fragile, so this module uses:

1. Vectorized seam compatibility for all oriented patches.
2. Beam search over the full grid.
3. Local rotation and swap refinement over the final assignment.

It intentionally has no heavy dependencies beyond NumPy and Pillow.
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image


ROTATIONS = (0, 1, 2, 3)
PATCH_NAME_RE = re.compile(r"^patch_(\d+)(?:_rot(0|90|180|270))?$", re.IGNORECASE)


@dataclass(frozen=True)
class SolverConfig:
    overlap_width: int | None = None
    edge_width: int = 3
    beam_width: int = 96
    beam_candidates: int = 18
    refine_passes: int = 3
    refine_candidates: int = 10
    swap_refine_passes: int = 5
    gradient_weight: float = 0.45
    boundary_weight: float = 0.12
    large_cost: float = 1.0e9
    rotations: tuple[int, ...] = (0,)
    anchor_rotations: tuple[int, ...] = (0,)
    rotation_penalty: float = 0.0
    block_refine_passes: int = 0
    exact_overlap_threshold: float = 1.0e-6


@dataclass
class Compatibility:
    states: list[tuple[int, int]]
    state_index: dict[tuple[int, int], int]
    piece_to_states: dict[int, list[int]]
    piece_ids: np.ndarray
    state_penalties: np.ndarray
    right_cost: np.ndarray
    down_cost: np.ndarray
    left_border: np.ndarray
    right_border: np.ndarray
    top_border: np.ndarray
    bottom_border: np.ndarray


def validate_patch_set(patches: dict[int, np.ndarray]) -> None:
    if 0 not in patches:
        raise ValueError("patch_0.png is required because it is the fixed top-left anchor")

    ids = sorted(patches)
    expected_ids = list(range(len(ids)))
    if ids != expected_ids:
        missing = sorted(set(expected_ids) - set(ids))
        extra = sorted(set(ids) - set(expected_ids))
        details = []
        if missing:
            details.append(f"missing ids: {missing[:10]}")
        if extra:
            details.append(f"unexpected ids: {extra[:10]}")
        raise ValueError("Patch ids must be contiguous from 0. " + "; ".join(details))

    first_shape = patches[0].shape
    if len(first_shape) != 3 or first_shape[2] != 3:
        raise ValueError(f"Patches must be RGB images, got shape {first_shape}")
    if first_shape[0] != first_shape[1]:
        raise ValueError(f"Patches must be square, got shape {first_shape}")
    if any(p.shape != first_shape for p in patches.values()):
        raise ValueError("All patches must have the same shape")


def load_patches(patches_dir: Path) -> dict[int, np.ndarray]:
    parsed_files = []
    for f in patches_dir.iterdir():
        if f.suffix.lower() != ".png":
            continue
        match = PATCH_NAME_RE.match(f.stem)
        if not match:
            continue
        pid = int(match.group(1))
        rotation_deg = int(match.group(2) or 0)
        parsed_files.append((pid, rotation_deg, f))

    files = sorted(parsed_files, key=lambda item: item[0])
    if not files:
        raise FileNotFoundError(f"No patch_*.png files found in {patches_dir}")

    raw_images: dict[int, np.ndarray] = {}
    rotation_by_id: dict[int, int] = {}
    rotation_tag_count = 0
    explicit_rotation_count = 0
    for idx, rotation_deg, f in files:
        raw_images[idx] = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        rotation_by_id[idx] = rotation_deg
        if PATCH_NAME_RE.match(f.stem).group(2) is not None:
            rotation_tag_count += 1
        if rotation_deg:
            explicit_rotation_count += 1

    if rotation_tag_count:
        convention_candidates = {
            "raw": dict(raw_images),
            "direct": {
                idx: np.rot90(raw_images[idx], (rotation_by_id[idx] // 90) % 4)
                for idx in raw_images
            },
            "inverse": {
                idx: np.rot90(raw_images[idx], (-rotation_by_id[idx] // 90) % 4)
                for idx in raw_images
            },
        }
        scored = []
        for convention_name, candidate_patches in convention_candidates.items():
            score = quick_reconstruction_score(candidate_patches)
            scored.append((score, convention_name, candidate_patches))
            print(f"[load] Convention {convention_name:7s} -> quick score {score:.4f}")
        scored.sort(key=lambda item: item[0])
        _, rotation_convention, patches = scored[0]
    else:
        patches = dict(raw_images)
        rotation_convention = "none"

    validate_patch_set(patches)
    first_shape = patches[0].shape

    print(f"[load] {len(patches)} patches | shape: {first_shape}")
    if explicit_rotation_count:
        print(
            f"[load] Auto-normalized {explicit_rotation_count} filename-encoded rotations "
            f"using the {rotation_convention} convention"
        )
    return patches


def quick_reconstruction_score(patches: dict[int, np.ndarray]) -> float:
    n = len(patches)
    grid_size = int(round(n**0.5))
    if grid_size * grid_size != n:
        return float("inf")

    patch_size = next(iter(patches.values())).shape[0]
    overlap_candidates = [0]
    try:
        estimated = estimate_overlap_width(
            patches,
            candidates=tuple(range(4, max(5, patch_size // 2 + 1), 4)),
            rotations=(0,),
        )
        overlap_candidates.extend(
            width
            for width in (estimated - 4, estimated, estimated + 4)
            if 0 < width < patch_size
        )
    except Exception:
        pass

    best = float("inf")
    for overlap_width in sorted(set(overlap_candidates)):
        cfg = SolverConfig(
            overlap_width=overlap_width,
            edge_width=1,
            gradient_weight=0.0,
            boundary_weight=0.8,
            beam_width=32,
            beam_candidates=8,
            refine_passes=0,
            refine_candidates=8,
            swap_refine_passes=0,
            block_refine_passes=0,
            rotations=(0,),
            anchor_rotations=(0,),
        )
        comp = build_compatibility(patches, cfg)
        exact_grid = exact_overlap_reconstruct(comp, grid_size, cfg)
        if exact_grid is not None:
            score = grid_seam_energy(exact_grid, comp)
        else:
            grid = frontier_reconstruct(comp, grid_size, cfg)
            score = grid_seam_energy(grid, comp)
        best = min(best, score)
    return best


def _strip_from_edge(img: np.ndarray, side: str, width: int) -> np.ndarray:
    """Return strip as (seam_length, width, channels), distance 0 at the edge."""
    if side == "left":
        return img[:, :width, :]
    if side == "right":
        return img[:, -1 : -width - 1 : -1, :]
    if side == "top":
        return np.transpose(img[:width, :, :], (1, 0, 2))
    if side == "bottom":
        return np.transpose(img[-1 : -width - 1 : -1, :, :], (1, 0, 2))
    raise ValueError(f"Unknown side: {side}")


def _edge_features(oriented: list[np.ndarray], side: str, width: int) -> tuple[np.ndarray, np.ndarray]:
    outer_parts = []
    inner_parts = []
    for img in oriented:
        strip = _strip_from_edge(img, side, width * 2).astype(np.float32) / 255.0
        outer_parts.append(strip[:, :width, :].reshape(-1))
        inner_parts.append(strip[:, width:, :].reshape(-1))
    return np.stack(outer_parts), np.stack(inner_parts)


def _overlap_features(oriented: list[np.ndarray], side: str, width: int) -> np.ndarray:
    parts = []
    for img in oriented:
        if side == "left":
            strip = img[:, :width, :]
        elif side == "right":
            strip = img[:, -width:, :]
        elif side == "top":
            strip = img[:width, :, :]
        elif side == "bottom":
            strip = img[-width:, :, :]
        else:
            raise ValueError(f"Unknown side: {side}")
        parts.append(strip.astype(np.float32).reshape(-1) / 255.0)
    return np.stack(parts)


def _overlap_pair_cost(a_side: np.ndarray, b_side: np.ndarray, batch: int = 32) -> np.ndarray:
    n = a_side.shape[0]
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        out[start:end] = np.mean(np.abs(a_side[start:end, None, :] - b_side[None, :, :]), axis=2)
    return out


def estimate_overlap_width(
    patches: dict[int, np.ndarray],
    candidates: tuple[int, ...] | None = None,
    rotations: tuple[int, ...] = ROTATIONS,
) -> int:
    """Estimate overlap by looking for sharp low-error matches from patch_0."""
    if candidates is None:
        patch_size = next(iter(patches.values())).shape[0]
        candidates = tuple(range(8, max(9, patch_size // 2 + 1), 4))

    anchor = patches[0].astype(np.float32) / 255.0
    others = [
        (pid, rot, np.rot90(p, rot).astype(np.float32) / 255.0)
        for pid, p in patches.items()
        if pid != 0
        for rot in rotations
    ]
    best: list[tuple[float, int, int, int]] = []
    for width in candidates:
        if width <= 0 or width >= anchor.shape[0]:
            continue
        right_scores = sorted(
            float(np.mean(np.abs(anchor[:, -width:, :] - p[:, :width, :])))
            for _, _, p in others
        )
        down_scores = sorted(
            float(np.mean(np.abs(anchor[-width:, :, :] - p[:width, :, :])))
            for _, _, p in others
        )
        right_min = right_scores[0]
        down_min = down_scores[0]
        right_gap = right_scores[1] - right_scores[0] if len(right_scores) > 1 else 0.0
        down_gap = down_scores[1] - down_scores[0] if len(down_scores) > 1 else 0.0
        score = right_min + down_min - 0.05 * (right_gap + down_gap)
        best.append((score, width, int(round(right_min * 1_000_000)), int(round(down_min * 1_000_000))))

    if not best:
        raise ValueError("Could not estimate overlap width")

    score, width, right_micro, down_micro = min(best, key=lambda x: x[0])
    print(
        f"[overlap] Estimated overlap {width}px "
        f"(patch_0 right/down errors {right_micro / 1_000_000:.6f}/{down_micro / 1_000_000:.6f})"
    )
    return width


def estimate_anchor_fit(
    patches: dict[int, np.ndarray],
    rotations: tuple[int, ...],
    overlap_width: int | None = None,
    anchor_rotations: tuple[int, ...] = (0,),
) -> tuple[float, int, int]:
    """Return (fit_score, overlap_width, best_anchor_rotation). Lower is better."""
    if overlap_width is None:
        overlap_width = estimate_overlap_width(patches, rotations=rotations)

    anchor_candidates = {
        rot: np.rot90(patches[0], rot).astype(np.float32) / 255.0
        for rot in anchor_rotations
    }
    others = [
        np.rot90(patch, rot).astype(np.float32) / 255.0
        for pid, patch in patches.items()
        if pid != 0
        for rot in rotations
    ]

    best_score = float("inf")
    best_anchor_rot = 0
    for anchor_rot, anchor in anchor_candidates.items():
        right_min = min(float(np.mean(np.abs(anchor[:, -overlap_width:, :] - other[:, :overlap_width, :]))) for other in others)
        down_min = min(float(np.mean(np.abs(anchor[-overlap_width:, :, :] - other[:overlap_width, :, :]))) for other in others)
        score = right_min + down_min
        if score < best_score:
            best_score = score
            best_anchor_rot = anchor_rot

    return best_score, overlap_width, best_anchor_rot


def _pair_cost(
    a_outer: np.ndarray,
    a_inner: np.ndarray,
    b_outer: np.ndarray,
    b_inner: np.ndarray,
    gradient_weight: float,
    batch: int = 32,
) -> np.ndarray:
    """Cost for side A touching opposite side B, lower is better."""
    n = a_outer.shape[0]
    out = np.empty((n, n), dtype=np.float32)
    a_grad = a_outer - a_inner
    b_grad_across_seam = b_inner - b_outer

    for start in range(0, n, batch):
        end = min(start + batch, n)
        boundary = np.mean(np.abs(a_outer[start:end, None, :] - b_outer[None, :, :]), axis=2)
        gradient = np.mean(
            np.abs(a_grad[start:end, None, :] - b_grad_across_seam[None, :, :]),
            axis=2,
        )
        out[start:end] = boundary + gradient_weight * gradient
    return out


def build_compatibility(patches: dict[int, np.ndarray], cfg: SolverConfig) -> Compatibility:
    states: list[tuple[int, int]] = []
    oriented: list[np.ndarray] = []
    piece_to_states: dict[int, list[int]] = {}

    for pid in sorted(patches):
        for rot in cfg.rotations:
            state_id = len(states)
            states.append((pid, rot))
            oriented.append(np.rot90(patches[pid], rot))
            piece_to_states.setdefault(pid, []).append(state_id)

    piece_ids = np.asarray([pid for pid, _ in states], dtype=np.int16)
    state_penalties = np.asarray(
        [cfg.rotation_penalty * float(rot != 0) for _, rot in states],
        dtype=np.float32,
    )
    state_index = {state: i for i, state in enumerate(states)}

    overlap_width = cfg.overlap_width
    if overlap_width is None:
        overlap_width = estimate_overlap_width(patches, rotations=cfg.rotations)

    print(f"[compat] Building compatibility scores for {len(states)} oriented states ...")
    t0 = time.time()
    if overlap_width and overlap_width > 0:
        r_overlap = _overlap_features(oriented, "right", overlap_width)
        l_overlap = _overlap_features(oriented, "left", overlap_width)
        b_overlap = _overlap_features(oriented, "bottom", overlap_width)
        t_overlap = _overlap_features(oriented, "top", overlap_width)
        right_cost = _overlap_pair_cost(r_overlap, l_overlap)
        down_cost = _overlap_pair_cost(b_overlap, t_overlap)
        print(f"[compat] Using {overlap_width}px overlap matching")
    else:
        r_outer, r_inner = _edge_features(oriented, "right", cfg.edge_width)
        l_outer, l_inner = _edge_features(oriented, "left", cfg.edge_width)
        b_outer, b_inner = _edge_features(oriented, "bottom", cfg.edge_width)
        t_outer, t_inner = _edge_features(oriented, "top", cfg.edge_width)
        right_cost = _pair_cost(r_outer, r_inner, l_outer, l_inner, cfg.gradient_weight)
        down_cost = _pair_cost(b_outer, b_inner, t_outer, t_inner, cfg.gradient_weight)
        print(f"[compat] Using {cfg.edge_width}px seam matching")

    for pid, state_ids in piece_to_states.items():
        right_cost[np.ix_(state_ids, state_ids)] = cfg.large_cost
        down_cost[np.ix_(state_ids, state_ids)] = cfg.large_cost

    left_border = np.min(right_cost, axis=0)
    right_border = np.min(right_cost, axis=1)
    top_border = np.min(down_cost, axis=0)
    bottom_border = np.min(down_cost, axis=1)
    print(f"[compat] Done in {time.time() - t0:.1f}s")

    return Compatibility(
        states=states,
        state_index=state_index,
        piece_to_states=piece_to_states,
        piece_ids=piece_ids,
        state_penalties=state_penalties,
        right_cost=right_cost,
        down_cost=down_cost,
        left_border=left_border,
        right_border=right_border,
        top_border=top_border,
        bottom_border=bottom_border,
    )


def _cell_boundary_bonus(state, row: int, col: int, grid_size: int, comp: Compatibility, cfg: SolverConfig):
    bonus = 0.0
    if col == 0:
        bonus -= cfg.boundary_weight * comp.left_border[state]
    if col == grid_size - 1:
        bonus -= cfg.boundary_weight * comp.right_border[state]
    if row == 0:
        bonus -= cfg.boundary_weight * comp.top_border[state]
    if row == grid_size - 1:
        bonus -= cfg.boundary_weight * comp.bottom_border[state]
    return bonus


def _mask_used(scores: np.ndarray, used: frozenset[int], comp: Compatibility) -> np.ndarray:
    masked = scores.copy()
    for pid in used:
        masked[comp.piece_to_states[pid]] = np.inf
    return masked


def _top_states(scores: np.ndarray, k: int) -> list[int]:
    finite = np.isfinite(scores)
    count = int(np.count_nonzero(finite))
    if count == 0:
        return []
    k = min(k, count)
    idx = np.argpartition(scores, k - 1)[:k]
    idx = idx[np.argsort(scores[idx])]
    return [int(i) for i in idx if np.isfinite(scores[i])]


def _state_penalty(state: int, comp: Compatibility, cfg: SolverConfig) -> float:
    return float(comp.state_penalties[state])


def _state_grid(flat: tuple[int, ...], grid_size: int) -> list[list[int]]:
    return [list(flat[r * grid_size : (r + 1) * grid_size]) for r in range(grid_size)]


def exact_overlap_reconstruct(comp: Compatibility, grid_size: int, cfg: SolverConfig) -> list[list[tuple[int, int]]] | None:
    """Recover the grid directly when overlap matches are exact or near-exact."""
    if len({pid for pid, _ in comp.states}) != len(comp.states):
        return None

    n = len(comp.states)
    threshold = cfg.exact_overlap_threshold
    pid_of_state = [pid for pid, _ in comp.states]

    right_cands = {pid: set() for pid in pid_of_state}
    down_cands = {pid: set() for pid in pid_of_state}
    left_of = {pid: set() for pid in pid_of_state}
    top_of = {pid: set() for pid in pid_of_state}

    for state in range(n):
        pid = pid_of_state[state]
        for other in np.where(comp.right_cost[state] <= threshold)[0]:
            other_pid = pid_of_state[int(other)]
            right_cands[pid].add(other_pid)
            left_of[other_pid].add(pid)
        for other in np.where(comp.down_cost[state] <= threshold)[0]:
            other_pid = pid_of_state[int(other)]
            down_cands[pid].add(other_pid)
            top_of[other_pid].add(pid)

    grid: list[list[int | None]] = [[None] * grid_size for _ in range(grid_size)]
    grid[0][0] = 0

    changed = True
    while changed:
        changed = False
        for row in range(grid_size):
            for col in range(grid_size):
                pid = grid[row][col]
                if pid is None:
                    continue
                if col + 1 < grid_size and grid[row][col + 1] is None and len(right_cands[pid]) == 1:
                    grid[row][col + 1] = next(iter(right_cands[pid]))
                    changed = True
                if row + 1 < grid_size and grid[row + 1][col] is None and len(down_cands[pid]) == 1:
                    grid[row + 1][col] = next(iter(down_cands[pid]))
                    changed = True

    used = {pid for row in grid for pid in row if pid is not None}
    remaining = set(pid_of_state) - used
    missing = [(row, col) for row in range(grid_size) for col in range(grid_size) if grid[row][col] is None]

    def cell_candidates(row: int, col: int) -> set[int]:
        cand = set(remaining)
        if col > 0 and grid[row][col - 1] is not None:
            cand &= right_cands[grid[row][col - 1]]
        if row > 0 and grid[row - 1][col] is not None:
            cand &= down_cands[grid[row - 1][col]]
        if col + 1 < grid_size and grid[row][col + 1] is not None:
            cand &= left_of[grid[row][col + 1]]
        if row + 1 < grid_size and grid[row + 1][col] is not None:
            cand &= top_of[grid[row + 1][col]]
        return cand

    solutions: list[list[list[int]]] = []

    def backtrack(open_cells: list[tuple[int, int]]) -> bool:
        if not open_cells:
            solutions.append([[int(pid) for pid in row] for row in grid])
            return len(solutions) >= 2

        best_pos = None
        best_cand = None
        for row, col in open_cells:
            cand = cell_candidates(row, col)
            if not cand:
                return False
            if best_cand is None or len(cand) < len(best_cand):
                best_pos = (row, col)
                best_cand = cand

        assert best_pos is not None and best_cand is not None
        row, col = best_pos
        rest = [cell for cell in open_cells if cell != best_pos]

        for pid in sorted(best_cand):
            grid[row][col] = pid
            remaining.remove(pid)
            if backtrack(rest):
                return True
            remaining.add(pid)
            grid[row][col] = None
        return False

    if missing:
        backtrack(missing)
    else:
        solutions.append([[int(pid) for pid in row] for row in grid])

    if len(solutions) != 1:
        return None

    solved = solutions[0]
    if any(len(row) != grid_size for row in solved):
        return None

    print(f"[exact] Recovered deterministic overlap grid with threshold {threshold:.1e}")
    return [[(pid, 0) for pid in row] for row in solved]


def _frontier_cell_scores(
    placements: dict[tuple[int, int], int],
    pos: tuple[int, int],
    grid_size: int,
    comp: Compatibility,
    cfg: SolverConfig,
) -> np.ndarray:
    row, col = pos
    scores = np.zeros(len(comp.states), dtype=np.float32)
    left = placements.get((row, col - 1))
    right = placements.get((row, col + 1))
    top = placements.get((row - 1, col))
    bottom = placements.get((row + 1, col))

    if left is not None:
        scores += comp.right_cost[left]
    if right is not None:
        scores += comp.right_cost[:, right]
    if top is not None:
        scores += comp.down_cost[top]
    if bottom is not None:
        scores += comp.down_cost[:, bottom]

    scores += _cell_boundary_bonus(np.arange(len(comp.states)), row, col, grid_size, comp, cfg)
    if cfg.rotation_penalty:
        scores += comp.state_penalties
    return scores


def frontier_reconstruct(comp: Compatibility, grid_size: int, cfg: SolverConfig) -> list[list[tuple[int, int]]]:
    anchor_states = [comp.state_index[(0, rot)] for rot in cfg.anchor_rotations if (0, rot) in comp.state_index]
    if not anchor_states:
        raise RuntimeError("No valid anchor states available for patch_0")

    best_flat = None
    best_energy = float("inf")

    for anchor_state in anchor_states:
        placements: dict[tuple[int, int], int] = {(0, 0): anchor_state}
        used = {0}
        frontier = {(0, 1), (1, 0)}

        while frontier:
            best_choice = None
            best_step = float("inf")

            for pos in frontier:
                row, col = pos
                if not (0 <= row < grid_size and 0 <= col < grid_size):
                    continue
                scores = _frontier_cell_scores(placements, pos, grid_size, comp, cfg)
                scores = _mask_used(scores, frozenset(used), comp)
                states = _top_states(scores, 1)
                if not states:
                    continue
                state = states[0]
                step = float(scores[state])
                if step < best_step:
                    best_step = step
                    best_choice = (pos, state)

            if best_choice is None:
                raise RuntimeError("Frontier growth failed to find a valid next placement")

            pos, state = best_choice
            placements[pos] = state
            frontier.remove(pos)
            pid, _ = comp.states[state]
            used.add(pid)

            row, col = pos
            for neighbor in ((row, col - 1), (row, col + 1), (row - 1, col), (row + 1, col)):
                nr, nc = neighbor
                if 0 <= nr < grid_size and 0 <= nc < grid_size and neighbor not in placements:
                    frontier.add(neighbor)

        flat = tuple(
            placements[(row, col)]
            for row in range(grid_size)
            for col in range(grid_size)
        )
        energy = _seam_energy(flat, grid_size, comp)
        if energy < best_energy:
            best_energy = energy
            best_flat = flat

    assert best_flat is not None
    print(f"[frontier] Initial seam energy: {best_energy:.3f}")
    refined = refine_assignment(best_flat, grid_size, comp, cfg)
    refined = swap_refine_assignment(refined, grid_size, comp, cfg)
    refined = block_refine_assignment(refined, grid_size, comp, cfg)
    return [[comp.states[state] for state in row] for row in _state_grid(refined, grid_size)]


def beam_reconstruct(comp: Compatibility, grid_size: int, cfg: SolverConfig) -> list[list[tuple[int, int]]]:
    anchor_states = [comp.state_index[(0, rot)] for rot in cfg.anchor_rotations if (0, rot) in comp.state_index]
    if not anchor_states:
        raise RuntimeError("No valid anchor states available for patch_0")
    beams: list[tuple[tuple[int, ...], frozenset[int], float]] = [
        ((anchor_state,), frozenset({0}), float(_cell_boundary_bonus(anchor_state, 0, 0, grid_size, comp, cfg)))
        for anchor_state in anchor_states
    ]
    total_cells = grid_size * grid_size
    t0 = time.time()

    for pos in range(1, total_cells):
        row, col = divmod(pos, grid_size)
        next_beams: list[tuple[tuple[int, ...], frozenset[int], float]] = []

        for flat, used, score in beams:
            candidate_score = np.zeros(len(comp.states), dtype=np.float32)
            if col > 0:
                left_state = flat[pos - 1]
                candidate_score += comp.right_cost[left_state]
            if row > 0:
                top_state = flat[pos - grid_size]
                candidate_score += comp.down_cost[top_state]
            if cfg.rotation_penalty:
                candidate_score += comp.state_penalties

            candidate_score = _mask_used(candidate_score, used, comp)
            for state in _top_states(candidate_score, cfg.beam_candidates):
                pid, _ = comp.states[state]
                step = float(candidate_score[state]) + _cell_boundary_bonus(state, row, col, grid_size, comp, cfg)
                next_beams.append((flat + (state,), used | frozenset({pid}), score + step))

        if not next_beams:
            raise RuntimeError(f"Beam search failed at cell {pos}")

        next_beams.sort(key=lambda x: x[2])
        beams = next_beams[: cfg.beam_width]
        if pos % grid_size == grid_size - 1:
            elapsed = time.time() - t0
            print(f"[beam] Row {row + 1:2d}/{grid_size} | {elapsed:.1f}s | best {beams[0][2]:.3f}")

    best_flat = beams[0][0]
    refined = refine_assignment(best_flat, grid_size, comp, cfg)
    refined = swap_refine_assignment(refined, grid_size, comp, cfg)
    refined = block_refine_assignment(refined, grid_size, comp, cfg)
    return [[comp.states[state] for state in row] for row in _state_grid(refined, grid_size)]


def _seam_energy(flat: tuple[int, ...], grid_size: int, comp: Compatibility, positions: set[int] | None = None) -> float:
    energy = 0.0
    for pos, state in enumerate(flat):
        row, col = divmod(pos, grid_size)
        if col < grid_size - 1:
            other = pos + 1
            if positions is None or pos in positions or other in positions:
                energy += float(comp.right_cost[state, flat[other]])
        if row < grid_size - 1:
            other = pos + grid_size
            if positions is None or pos in positions or other in positions:
                energy += float(comp.down_cost[state, flat[other]])
    return energy


def _affected(pos_a: int, pos_b: int, grid_size: int) -> set[int]:
    out: set[int] = set()
    for pos in (pos_a, pos_b):
        row, col = divmod(pos, grid_size)
        out.add(pos)
        if col > 0:
            out.add(pos - 1)
        if col < grid_size - 1:
            out.add(pos + 1)
        if row > 0:
            out.add(pos - grid_size)
        if row < grid_size - 1:
            out.add(pos + grid_size)
    return out


def _local_scores(flat: tuple[int, ...], pos: int, grid_size: int, comp: Compatibility, cfg: SolverConfig) -> np.ndarray:
    row, col = divmod(pos, grid_size)
    scores = np.zeros(len(comp.states), dtype=np.float32)
    if col > 0:
        scores += comp.right_cost[flat[pos - 1]]
    if col < grid_size - 1:
        scores += comp.right_cost[:, flat[pos + 1]]
    if row > 0:
        scores += comp.down_cost[flat[pos - grid_size]]
    if row < grid_size - 1:
        scores += comp.down_cost[:, flat[pos + grid_size]]
    scores += _cell_boundary_bonus(np.arange(len(comp.states)), row, col, grid_size, comp, cfg)
    if cfg.rotation_penalty:
        scores += comp.state_penalties
    return scores


def _best_rotation_for_piece(
    flat: tuple[int, ...],
    pos: int,
    pid: int,
    grid_size: int,
    comp: Compatibility,
    cfg: SolverConfig,
) -> int:
    scores = _local_scores(flat, pos, grid_size, comp, cfg)
    state_ids = comp.piece_to_states[pid]
    return min(state_ids, key=lambda state: float(scores[state]))


def refine_assignment(flat: tuple[int, ...], grid_size: int, comp: Compatibility, cfg: SolverConfig) -> tuple[int, ...]:
    current = list(flat)
    anchor = comp.state_index[(0, cfg.anchor_rotations[0])]
    print(f"[refine] Initial seam energy: {_seam_energy(tuple(current), grid_size, comp):.3f}")

    for pass_id in range(cfg.refine_passes):
        changed = 0

        for pos in range(1, len(current)):
            pid, _ = comp.states[current[pos]]
            best_state = _best_rotation_for_piece(tuple(current), pos, pid, grid_size, comp, cfg)
            if best_state != current[pos]:
                current[pos] = best_state
                changed += 1

        pid_to_pos = {comp.states[state][0]: pos for pos, state in enumerate(current)}
        for pos in range(1, len(current)):
            before_flat = tuple(current)
            before_state = current[pos]
            before_pid, _ = comp.states[before_state]

            scores = _local_scores(before_flat, pos, grid_size, comp, cfg)
            scores[anchor] = np.inf
            for state in _top_states(scores, cfg.refine_candidates):
                candidate_pid, _ = comp.states[state]
                swap_pos = pid_to_pos.get(candidate_pid)
                if swap_pos is None or swap_pos == pos or swap_pos == 0:
                    continue

                affected = _affected(pos, swap_pos, grid_size)
                old_energy = _seam_energy(before_flat, grid_size, comp, affected)

                best_trial = None
                best_energy = old_energy
                for state_a in comp.piece_to_states[candidate_pid]:
                    for state_b in comp.piece_to_states[before_pid]:
                        trial = list(before_flat)
                        trial[pos] = state_a
                        trial[swap_pos] = state_b
                        trial_energy = _seam_energy(tuple(trial), grid_size, comp, affected)
                        if trial_energy + 1.0e-7 < best_energy:
                            best_energy = trial_energy
                            best_trial = trial

                if best_trial is not None:
                    current = best_trial
                    pid_to_pos[before_pid] = swap_pos
                    pid_to_pos[candidate_pid] = pos
                    changed += 1
                    break

        energy = _seam_energy(tuple(current), grid_size, comp)
        print(f"[refine] Pass {pass_id + 1}/{cfg.refine_passes} | changes {changed} | energy {energy:.3f}")
        if changed == 0:
            break

    return tuple(current)


def swap_refine_assignment(
    flat: tuple[int, ...],
    grid_size: int,
    comp: Compatibility,
    cfg: SolverConfig,
) -> tuple[int, ...]:
    """Exhaustive 2-opt style tile swaps to reduce total seam energy."""
    if cfg.swap_refine_passes <= 0:
        return flat

    current = list(flat)
    print(f"[2opt] Initial seam energy: {_seam_energy(tuple(current), grid_size, comp):.3f}")

    for pass_id in range(cfg.swap_refine_passes):
        changes = 0
        gain = 0.0
        before_pass = tuple(current)

        for pos_a in range(1, len(current) - 1):
            for pos_b in range(pos_a + 1, len(current)):
                affected = _affected(pos_a, pos_b, grid_size)
                old_energy = _seam_energy(tuple(current), grid_size, comp, affected)

                original_state_a = current[pos_a]
                original_state_b = current[pos_b]
                pid_a, _ = comp.states[original_state_a]
                pid_b, _ = comp.states[original_state_b]
                best_state_a = original_state_a
                best_state_b = original_state_b
                best_energy = old_energy

                for state_a in comp.piece_to_states[pid_b]:
                    for state_b in comp.piece_to_states[pid_a]:
                        current[pos_a] = state_a
                        current[pos_b] = state_b
                        new_energy = _seam_energy(tuple(current), grid_size, comp, affected)
                        if new_energy + 1.0e-8 < best_energy:
                            best_energy = new_energy
                            best_state_a = state_a
                            best_state_b = state_b

                if best_energy + 1.0e-8 < old_energy:
                    current[pos_a] = best_state_a
                    current[pos_b] = best_state_b
                    changes += 1
                    gain += old_energy - best_energy
                else:
                    current[pos_a] = original_state_a
                    current[pos_b] = original_state_b

        total = _seam_energy(tuple(current), grid_size, comp)
        print(
            f"[2opt] Pass {pass_id + 1}/{cfg.swap_refine_passes} | "
            f"swaps {changes} | gain {gain:.3f} | energy {total:.3f}"
        )
        if changes == 0 or tuple(current) == before_pass:
            break

    return tuple(current)


def _window_positions(top: int, left: int, grid_size: int) -> tuple[list[int], set[int]]:
    cells = []
    affected: set[int] = set()
    for dr in (0, 1):
        for dc in (0, 1):
            pos = (top + dr) * grid_size + (left + dc)
            cells.append(pos)
            row, col = divmod(pos, grid_size)
            affected.add(pos)
            if col > 0:
                affected.add(pos - 1)
            if col < grid_size - 1:
                affected.add(pos + 1)
            if row > 0:
                affected.add(pos - grid_size)
            if row < grid_size - 1:
                affected.add(pos + grid_size)
    return cells, affected


def block_refine_assignment(
    flat: tuple[int, ...],
    grid_size: int,
    comp: Compatibility,
    cfg: SolverConfig,
) -> tuple[int, ...]:
    """Exhaustive 2x2 local repair over the current state assignments."""
    if cfg.block_refine_passes <= 0 or grid_size < 2:
        return flat

    current = list(flat)
    print(f"[block] Initial seam energy: {_seam_energy(tuple(current), grid_size, comp):.3f}")

    for pass_id in range(cfg.block_refine_passes):
        changes = 0
        for top in range(grid_size - 1):
            for left in range(grid_size - 1):
                cells, affected = _window_positions(top, left, grid_size)
                base_flat = tuple(current)
                base_energy = _seam_energy(base_flat, grid_size, comp, affected)
                states = [current[pos] for pos in cells]
                best_perm = states
                best_energy = base_energy

                for perm in permutations(states):
                    if perm == tuple(states):
                        continue
                    trial = list(base_flat)
                    for pos, state in zip(cells, perm):
                        trial[pos] = state
                    trial_energy = _seam_energy(tuple(trial), grid_size, comp, affected)
                    if trial_energy + 1.0e-8 < best_energy:
                        best_energy = trial_energy
                        best_perm = list(perm)

                if best_energy + 1.0e-8 < base_energy:
                    for pos, state in zip(cells, best_perm):
                        current[pos] = state
                    changes += 1

        total = _seam_energy(tuple(current), grid_size, comp)
        print(f"[block] Pass {pass_id + 1}/{cfg.block_refine_passes} | windows {changes} | energy {total:.3f}")
        if changes == 0:
            break

    return tuple(current)


def reconstruct_grid(patches: dict[int, np.ndarray], grid_size: int, cfg: SolverConfig | None = None) -> list[list[tuple[int, int]]]:
    cfg = cfg or SolverConfig()
    comp = build_compatibility(patches, cfg)
    exact_grid = exact_overlap_reconstruct(comp, grid_size, cfg)
    if exact_grid is not None:
        return exact_grid
    return frontier_reconstruct(comp, grid_size, cfg)


def grid_seam_energy(grid: list[list[tuple[int, int]]], comp: Compatibility) -> float:
    flat = tuple(comp.state_index[state] for row in grid for state in row)
    return _seam_energy(flat, len(grid), comp)


def stitch_map(
    patches: dict[int, np.ndarray],
    grid: list[list[tuple[int, int]]],
    output_path: Path,
    overlap_width: int | None = None,
) -> Image.Image:
    grid_size = len(grid)
    patch_h, patch_w = next(iter(patches.values())).shape[:2]
    if overlap_width is None:
        overlap_width = estimate_overlap_width(patches)
    stride_y = patch_h - overlap_width
    stride_x = patch_w - overlap_width
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError(f"Invalid overlap width {overlap_width} for patch size {(patch_w, patch_h)}")

    canvas_w = patch_w + (grid_size - 1) * stride_x
    canvas_h = patch_h + (grid_size - 1) * stride_y
    accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    for row in range(grid_size):
        for col in range(grid_size):
            pid, rot = grid[row][col]
            arr = np.rot90(patches[pid], rot).astype(np.float32)
            y = row * stride_y
            x = col * stride_x
            accum[y : y + patch_h, x : x + patch_w] += arr
            weight[y : y + patch_h, x : x + patch_w] += 1.0

    canvas_arr = np.clip(accum / np.maximum(weight, 1.0), 0, 255).astype(np.uint8)
    canvas = Image.fromarray(canvas_arr, "RGB")

    try:
        canvas.save(output_path)
        saved_path = output_path
    except PermissionError:
        saved_path = output_path.with_name(f"{output_path.stem}_updated{output_path.suffix}")
        canvas.save(saved_path)
        print(f"[stitch] Could not overwrite locked file: {output_path}")
    print(
        f"[stitch] Saved -> {saved_path} ({canvas.size[0]}x{canvas.size[1]} px, "
        f"overlap={overlap_width}px, stride={stride_x}px)"
    )
    return canvas
