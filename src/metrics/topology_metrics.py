"""COR / INF vessel-topology metrics.

Best-effort reimplementation of the RRWNet paper's definition (exact
organizer sampling protocol is not publicly documented -- calibrate against
the baseline's published Task1/Task2 scores before trusting absolute values;
see notebooks/validate_metrics_against_baseline.ipynb):

  - Sample point pairs on the GT vessel skeleton.
  - A path is "infeasible" (INF) if the two points are not connected in the
    predicted skeleton (snapped within `snap_radius` px).
  - A path is "correct" (COR) if they ARE connected in the prediction and the
    shortest-path length differs from the GT shortest-path length by <10%.
  - COR and INF are independent fractions over the sampled pairs (a pair can
    be neither: connected but off by >=10%).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def _skeleton_graph(binary_mask: np.ndarray):
    """8-connected pixel graph over a skeletonized binary mask.

    Returns (coords Nx2 array of (row,col), sparse adjacency csr_matrix with
    Euclidean edge weights).
    """
    skel = skeletonize(binary_mask.astype(bool))
    ys, xs = np.nonzero(skel)
    coords = np.stack([ys, xs], axis=1)
    n = len(coords)
    if n == 0:
        return coords, coo_matrix((0, 0)).tocsr()

    index_of = -np.ones(skel.shape, dtype=np.int64)
    index_of[ys, xs] = np.arange(n)

    rows, cols, weights = [], [], []
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dy, dx in offsets:
        ny, nx = ys + dy, xs + dx
        valid = (ny >= 0) & (ny < skel.shape[0]) & (nx >= 0) & (nx < skel.shape[1])
        neighbor_idx = np.full(n, -1, dtype=np.int64)
        neighbor_idx[valid] = index_of[ny[valid], nx[valid]]
        has_edge = neighbor_idx >= 0
        src = np.arange(n)[has_edge]
        dst = neighbor_idx[has_edge]
        w = np.hypot(dy, dx)
        rows.append(src)
        cols.append(dst)
        weights.append(np.full(src.shape, w))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    weights = np.concatenate(weights)
    adj = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    return coords, adj


def _sample_pairs(
    coords: np.ndarray,
    labels: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
    min_dist: float = 20.0,
):
    """Sample n_pairs (i, j) index pairs, restricted to the same connected
    component (a GT "path" only exists within one component) and at least
    min_dist px apart (avoids trivial near-duplicate pairs)."""
    from collections import defaultdict

    by_label = defaultdict(list)
    for idx, lbl in enumerate(labels):
        by_label[lbl].append(idx)
    # Only components large enough to contain a meaningful path.
    eligible_components = [idxs for idxs in by_label.values() if len(idxs) >= 5]
    if not eligible_components:
        return []
    weights = np.array([len(idxs) for idxs in eligible_components], dtype=np.float64)
    weights /= weights.sum()

    pairs = []
    attempts = 0
    max_attempts = n_pairs * 50
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        comp = eligible_components[rng.choice(len(eligible_components), p=weights)]
        i, j = rng.choice(comp, size=2, replace=False)
        if np.hypot(*(coords[i] - coords[j])) < min_dist:
            continue
        pairs.append((i, j))
    return pairs


def compute_cor_inf(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    n_pairs: int = 500,
    snap_radius: float = 5.0,
    tolerance: float = 0.10,
    seed: int = 0,
) -> dict:
    """Returns {'COR': float, 'INF': float, 'n_pairs_evaluated': int}."""
    rng = np.random.default_rng(seed)

    gt_coords, gt_adj = _skeleton_graph(gt_mask)
    pred_coords, pred_adj = _skeleton_graph(pred_mask)

    if len(gt_coords) < 2:
        return {"COR": 0.0, "INF": 1.0, "n_pairs_evaluated": 0}
    if len(pred_coords) == 0:
        return {"COR": 0.0, "INF": 1.0, "n_pairs_evaluated": n_pairs}

    _, gt_labels = connected_components(gt_adj, directed=False)
    pairs = _sample_pairs(gt_coords, gt_labels, n_pairs, rng)
    pred_tree = cKDTree(pred_coords)

    n_correct = 0
    n_infeasible = 0
    n_evaluated = 0

    # Group by source to reuse a single Dijkstra call per unique source index.
    from collections import defaultdict

    by_source = defaultdict(list)
    for i, j in pairs:
        by_source[i].append(j)

    for i, targets in by_source.items():
        gt_dist = shortest_path(gt_adj, method="D", indices=i, directed=False)

        snap_i_d, snap_i_idx = pred_tree.query(gt_coords[i])
        if snap_i_d > snap_radius:
            n_infeasible += len(targets)
            n_evaluated += len(targets)
            continue
        pred_dist = shortest_path(pred_adj, method="D", indices=snap_i_idx, directed=False)

        for j in targets:
            n_evaluated += 1
            gt_len = gt_dist[j]
            if not np.isfinite(gt_len):
                n_evaluated -= 1
                continue

            snap_j_d, snap_j_idx = pred_tree.query(gt_coords[j])
            if snap_j_d > snap_radius:
                n_infeasible += 1
                continue

            pred_len = pred_dist[snap_j_idx]
            if not np.isfinite(pred_len):
                n_infeasible += 1
                continue

            if abs(pred_len - gt_len) / gt_len < tolerance:
                n_correct += 1

    if n_evaluated == 0:
        return {"COR": 0.0, "INF": 1.0, "n_pairs_evaluated": 0}

    return {
        "COR": n_correct / n_evaluated,
        "INF": n_infeasible / n_evaluated,
        "n_pairs_evaluated": n_evaluated,
    }
