"""
Poincaré Ball Operations for PMIS v2.

Implements hyperbolic geometry for hierarchical memory representation.
Includes: distance, Möbius addition, exp/log maps, coordinate assignment,
and relation-specific transformations (MuRP-inspired).
"""

import numpy as np
from typing import Optional, Dict

EPS = 1e-7


def poincare_distance(u: np.ndarray, v: np.ndarray, c: float = 1.0) -> float:
    """
    Geodesic distance between two points in the Poincaré ball.
    d(u,v) = (2/√c) × arctanh(√c × ||(-u) ⊕ v||)
    """
    sqrt_c = np.sqrt(c)
    mob = mobius_addition(-u, v, c)
    norm_mob = np.clip(np.linalg.norm(mob), 0, 1.0 - EPS)
    return (2.0 / sqrt_c) * np.arctanh(sqrt_c * norm_mob)


def mobius_addition(u: np.ndarray, v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Möbius addition: u ⊕ v in the Poincaré ball.
    """
    u_dot_v = np.dot(u, v)
    u_norm_sq = np.dot(u, u)
    v_norm_sq = np.dot(v, v)

    numerator = ((1 + 2 * c * u_dot_v + c * v_norm_sq) * u +
                 (1 - c * u_norm_sq) * v)
    denominator = 1 + 2 * c * u_dot_v + (c ** 2) * u_norm_sq * v_norm_sq

    result = numerator / (denominator + EPS)
    return project_to_ball(result)


def project_to_ball(x: np.ndarray, max_norm: float = 1.0 - EPS) -> np.ndarray:
    """Project point back into the open unit ball."""
    norm = np.linalg.norm(x)
    if norm >= max_norm:
        return x * (max_norm / (norm + EPS))
    return x


def exp_map_origin(v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Exponential map from tangent space at origin → Poincaré ball."""
    sqrt_c = np.sqrt(c)
    v_norm = np.linalg.norm(v)
    if v_norm < EPS:
        return np.zeros_like(v)
    return np.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm + EPS)


def log_map_origin(y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Logarithmic map from Poincaré ball → tangent space at origin."""
    sqrt_c = np.sqrt(c)
    y_norm = np.clip(np.linalg.norm(y), 0, 1.0 - EPS)
    if y_norm < EPS:
        return np.zeros_like(y)
    return np.arctanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm + EPS)


def hierarchy_level(point: np.ndarray) -> float:
    """Extract hierarchy level from norm. 0=abstract (origin), 1=specific (boundary)."""
    return float(np.linalg.norm(point))


# ---------------------------------------------------------------------------
# Coordinate assignment for new memory nodes
# ---------------------------------------------------------------------------

class ProjectionManager:
    """
    Manages the random projection matrix from euclidean (1536d) to
    hyperbolic (32d) space. The matrix is seeded for reproducibility
    and persisted to disk so coordinates remain consistent across runs.
    """

    def __init__(self, input_dim: int = 1536, output_dim: int = 32, seed: int = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        self._matrix: Optional[np.ndarray] = None

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            rng = np.random.RandomState(self.seed)
            self._matrix = rng.randn(self.output_dim, self.input_dim)
            # Normalize rows
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            self._matrix = self._matrix / (norms + EPS)
        return self._matrix

    def save(self, path: str):
        np.save(path, self.matrix)

    def load(self, path: str):
        self._matrix = np.load(path)
        assert self._matrix.shape == (self.output_dim, self.input_dim), \
            f"Loaded matrix shape {self._matrix.shape} doesn't match expected ({self.output_dim}, {self.input_dim})"

    def project(self, euclidean_embedding: np.ndarray) -> np.ndarray:
        """Project a euclidean embedding down to hyperbolic dimensionality."""
        projected = self.matrix @ euclidean_embedding
        norm = np.linalg.norm(projected)
        if norm > EPS:
            projected = projected / norm
        return projected


def assign_hyperbolic_coords(
    euclidean_embedding: np.ndarray,
    level: str,
    projection_manager: ProjectionManager,
    parent_coords: Optional[np.ndarray] = None,
    hyperparams: Optional[Dict] = None,
) -> np.ndarray:
    """
    Assign Poincaré ball coordinates to a new memory node.

    1. Project euclidean → low-dim direction vector
    2. Scale norm based on hierarchy level
    3. If parent exists, bias direction toward parent
    """
    hp = hyperparams or {}

    # Step 1: Get direction via projection
    direction = projection_manager.project(euclidean_embedding)

    # Step 2: Determine target norm from level
    norm_ranges = {
        "SC":  (hp.get("poincare_sc_norm_min", 0.05),  hp.get("poincare_sc_norm_max", 0.20)),
        "CTX": (hp.get("poincare_ctx_norm_min", 0.35), hp.get("poincare_ctx_norm_max", 0.60)),
        "ANC": (hp.get("poincare_anc_norm_min", 0.70), hp.get("poincare_anc_norm_max", 0.95)),
    }
    lo, hi = norm_ranges.get(level, (0.35, 0.60))
    target_norm = np.random.uniform(lo, hi)

    # Step 3: If parent exists, blend direction toward parent
    if parent_coords is not None and np.linalg.norm(parent_coords) > EPS:
        parent_dir = parent_coords / (np.linalg.norm(parent_coords) + EPS)
        dim = min(len(direction), len(parent_dir))
        blended = 0.6 * direction[:dim] + 0.4 * parent_dir[:dim]
        blended = blended / (np.linalg.norm(blended) + EPS)
        # Pad if needed
        if len(blended) < len(direction):
            blended = np.concatenate([blended, direction[dim:]])
        direction = blended

    coords = direction * target_norm
    return project_to_ball(coords)


def place_near_parent(
    euclidean_embedding: np.ndarray,
    parent_coords: np.ndarray,
    level: str,
    projection_manager: "ProjectionManager",
    hyperparams: Optional[Dict] = None,
) -> np.ndarray:
    """
    Place a new node NEAR its parent using exponential map.

    Instead of random projection, this uses the parent's position
    to place the child nearby but slightly further from origin,
    with angular diversity from the euclidean content.

    This gives meaningful initial positions that RSGD can refine.
    """
    hp = hyperparams or {}

    # Get content-derived direction for angular diversity
    content_direction = projection_manager.project(euclidean_embedding)

    # Map parent to tangent space at origin
    parent_tangent = log_map_origin(parent_coords)
    parent_norm = np.linalg.norm(parent_tangent)

    if parent_norm < EPS:
        # Parent is at origin — fall back to regular assignment
        return assign_hyperbolic_coords(euclidean_embedding, level, projection_manager, None, hp)

    # Child should be slightly further from origin than parent
    # Push outward along parent's direction + angular offset from content
    outward_scale = 1.3 if level == "ANC" else 1.15
    offset = parent_tangent * outward_scale

    # Add angular perturbation from content (for diversity among siblings)
    perturbation_scale = 0.2
    dim = min(len(content_direction), len(offset))
    offset[:dim] += content_direction[:dim] * perturbation_scale

    # Map back to Poincare ball
    coords = exp_map_origin(offset)
    return project_to_ball(coords, max_norm=hp.get("rsgd_max_norm", 0.95))


# ---------------------------------------------------------------------------
# Relation-specific transformations (MuRP-inspired)
# ---------------------------------------------------------------------------

class RelationTransform:
    """
    Applies relation-specific transformations in hyperbolic space.

    Each relation type has a learned translation vector and diagonal scaling.
    This is a simplified MuRP: for each relation r, we transform a point x as:
        x_r = diag(s_r) ⊗ x ⊕ t_r

    where s_r is a per-relation scale and t_r is a per-relation translation,
    both in the tangent space at origin.
    """

    def __init__(self, dim: int = 32, seed: int = 123):
        self.dim = dim
        self.transforms: Dict[str, Dict[str, np.ndarray]] = {}
        self._rng = np.random.RandomState(seed)

    def register_relation(self, relation_id: str):
        """Register a new relation type with random initial parameters."""
        if relation_id not in self.transforms:
            self.transforms[relation_id] = {
                "scale": np.ones(self.dim) + self._rng.randn(self.dim) * 0.1,
                "translation": self._rng.randn(self.dim) * 0.05,
            }

    def transform(self, point: np.ndarray, relation_id: str) -> np.ndarray:
        """
        Apply relation-specific transformation.

        point: coordinates in Poincaré ball
        relation_id: which relation/tree lens to apply
        Returns: transformed point still in the Poincaré ball
        """
        if relation_id not in self.transforms:
            self.register_relation(relation_id)

        params = self.transforms[relation_id]

        # Map to tangent space
        tangent = log_map_origin(point)

        # Apply scale + translation in tangent space
        tangent = tangent * params["scale"] + params["translation"]

        # Map back to ball
        result = exp_map_origin(tangent)
        return project_to_ball(result)

    def save(self, path: str):
        """Save all relation parameters."""
        data = {}
        for rel_id, params in self.transforms.items():
            data[rel_id] = {
                "scale": params["scale"].tolist(),
                "translation": params["translation"].tolist(),
            }
        np.save(path, data, allow_pickle=True)

    def load(self, path: str):
        """Load relation parameters."""
        data = np.load(path, allow_pickle=True).item()
        for rel_id, params in data.items():
            self.transforms[rel_id] = {
                "scale": np.array(params["scale"]),
                "translation": np.array(params["translation"]),
            }
