"""
Mission 01 — Data Foundations: Point Clouds, Meshes, and Voxels
================================================================

End-to-end preprocessing pipeline that consolidates every step of the lesson:

    1. Load raw scan (or synthesize a fallback indoor room)
    2. Compute spatial statistics and center coordinates
    3. Voxel-downsample for uniform spatial density
    4. Estimate consistently-oriented surface normals
    5. Reconstruct a triangle mesh with Ball Pivoting
    6. Build a regular voxel grid for CNN-style consumers
    7. Compute the planarity geometric feature (eigenvalue analysis)
    8. Project sampled features back to full resolution via KDTree
    9. Export a binary little-endian PLY with the scalar field
   10. (optional) Visualize the outputs
   11. Print a profiling summary

Run:
    python mission_01_data_foundations.py            # console + files + one window per output
    python mission_01_data_foundations.py --no-viz   # headless: console + files only

In an IDE (e.g. Spyder) the output windows open automatically (VISUALIZE = True below).

Requirements:
    pip install numpy open3d scipy
    pip install matplotlib            # only needed for --viz
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ============================================================================
# Configuration
# ============================================================================
# Real lesson scan — resolved relative to THIS script (not the working
# directory) so it loads no matter where the file is launched from. The scan
# lives in the SpatialOS lesson tree; if it's missing, load_pc() falls back
# to a synthetic room.
try:
    _BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:                      # __file__ undefined (e.g. pasted into a console)
    _BASE_DIR = Path.cwd().parent
INPUT_PATH       = _BASE_DIR / "SpatialOS" / "01-data-foundations" / "DATA" / "indoor_room_labeled.ply"
RESULTS_DIR      = Path("../RESULTS/mission_01")
VOXEL_SIZE_SAMPLE = 0.05   # 5 cm — downsampling resolution
VOXEL_SIZE_GRID   = 0.10   # 10 cm — voxel grid resolution
K_NORMALS         = 20     # neighbors for normal estimation
K_PLANARITY       = 15     # neighbors for planarity feature
ORIENT_K          = 10     # k for orient_normals_consistent_tangent_plane

VISUALIZE         = True      # open a window per output by default; set False (or --no-viz) for headless
CMAP_NAME         = "viridis" # colormap for the planarity-colored cloud / histogram


# ============================================================================
# 1. Point cloud loading (with synthetic fallback)
# ============================================================================
def _synthetic_room(n_points: int = 60_000, seed: int = 42) -> np.ndarray:
    """Generate a synthetic 10x10x3 m indoor room: floor, ceiling, 4 walls.

    Used as a fallback when the real scan file is not available so the
    pipeline remains end-to-end runnable in any environment.
    """
    rng = np.random.default_rng(seed)
    n_per = n_points // 6
    noise = lambda n: rng.normal(0.0, 0.005, n)   # ~5 mm scanner noise

    surfaces = [
        # Floor (z = 0)
        np.column_stack([rng.uniform(0, 10, n_per),
                         rng.uniform(0, 10, n_per),
                         noise(n_per)]),
        # Ceiling (z = 3)
        np.column_stack([rng.uniform(0, 10, n_per),
                         rng.uniform(0, 10, n_per),
                         3.0 + noise(n_per)]),
        # Wall x = 0
        np.column_stack([noise(n_per),
                         rng.uniform(0, 10, n_per),
                         rng.uniform(0, 3, n_per)]),
        # Wall x = 10
        np.column_stack([10.0 + noise(n_per),
                         rng.uniform(0, 10, n_per),
                         rng.uniform(0, 3, n_per)]),
        # Wall y = 0
        np.column_stack([rng.uniform(0, 10, n_per),
                         noise(n_per),
                         rng.uniform(0, 3, n_per)]),
        # Wall y = 10
        np.column_stack([rng.uniform(0, 10, n_per),
                         10.0 + noise(n_per),
                         rng.uniform(0, 3, n_per)]),
    ]
    return np.vstack(surfaces).astype(np.float32)


def load_pc(path: str | Path) -> o3d.geometry.PointCloud:
    """Load a point cloud from disk, or synthesise a scene if the file is absent."""
    path = Path(path)
    pcd = o3d.geometry.PointCloud()

    if path.exists():
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) > 0:
            return pcd
        print(f"[!] {path} loaded but contains no points — falling back to synthetic data.")
    else:
        print(f"[!] {path} not found — generating synthetic indoor room.")

    pts = _synthetic_room()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


# ============================================================================
# 2. Spatial statistics & centering
# ============================================================================
def compute_spatial_stats(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (min, max, mean) per axis. Use mean as the centering offset."""
    p_min  = pts.min(axis=0)
    p_max  = pts.max(axis=0)
    center = pts.mean(axis=0)
    return p_min, p_max, center


# ============================================================================
# 3. Voxel downsampling
# ============================================================================
def smart_voxel_sample(pts: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    """Spatially uniform downsampling — one point per voxel."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_down.points)


# ============================================================================
# 4. Surface normal estimation
# ============================================================================
def estimate_normals(pts: np.ndarray, k: int = 20) -> o3d.geometry.PointCloud:
    """Estimate per-point normals via local PCA, then orient consistently."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    # Propagate orientation across the surface so normals don't flip mid-wall.
    pcd.orient_normals_consistent_tangent_plane(ORIENT_K)
    return pcd


# ============================================================================
# 5. Mesh reconstruction (Ball Pivoting)
# ============================================================================
def reconstruct_mesh(pts: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Build a triangle mesh from points using the Ball Pivoting Algorithm.

    Two radii are used so that fine detail and small gaps are both handled.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals()

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist  = float(np.mean(distances))
    radii     = [avg_dist, avg_dist * 2.0]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh.compute_vertex_normals()
    return mesh


# ============================================================================
# 6. Voxelization
# ============================================================================
def voxelize_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.1) -> o3d.geometry.VoxelGrid:
    """Convert a point cloud into a regular VoxelGrid (occupancy)."""
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)


# ============================================================================
# 7. Geometric feature: planarity
# ============================================================================
def compute_planarity(pts: np.ndarray, k: int = 15) -> np.ndarray:
    """Local planarity from the eigenvalues of each k-neighborhood's covariance.

    For ascending eigenvalues (l0 <= l1 <= l2):
        planarity = (l1 - l0) / l2
    Values close to 1 = flat surface; close to 0 = linear/scattered/corner.
    """
    n = len(pts)
    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=k, workers=-1)

    planarity = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neighbors = pts[indices[i]]
        cov = np.cov(neighbors.T)
        # eigvalsh returns ascending eigenvalues for symmetric matrices.
        l = np.linalg.eigvalsh(cov)
        planarity[i] = (l[1] - l[0]) / (l[2] + 1e-8)

    return planarity


# Bonus features from Exercise 2 — included so the script doubles as a reference.
def compute_linearity(pts: np.ndarray, k: int = 15) -> np.ndarray:
    """linearity = (l2 - l1) / l2  — highlights edges, beams, wires."""
    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=k, workers=-1)
    out = np.zeros(len(pts), dtype=np.float32)
    for i in range(len(pts)):
        l = np.linalg.eigvalsh(np.cov(pts[indices[i]].T))
        out[i] = (l[2] - l[1]) / (l[2] + 1e-8)
    return out


def compute_sphericity(pts: np.ndarray, k: int = 15) -> np.ndarray:
    """sphericity = l0 / l2 — highlights corners, clutter, noise."""
    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=k, workers=-1)
    out = np.zeros(len(pts), dtype=np.float32)
    for i in range(len(pts)):
        l = np.linalg.eigvalsh(np.cov(pts[indices[i]].T))
        out[i] = l[0] / (l[2] + 1e-8)
    return out


# ============================================================================
# 8. Full-scale projection via KDTree
# ============================================================================
def project_scalars(pts_full: np.ndarray,
                    pts_sampled: np.ndarray,
                    scalars_sampled: np.ndarray) -> np.ndarray:
    """Transfer per-point scalars from a sampled cloud back to the full cloud.

    Each full-res point inherits the value of its nearest sampled neighbor.
    """
    tree = cKDTree(pts_sampled)
    _, idx = tree.query(pts_full, k=1, workers=-1)
    return scalars_sampled[idx]


# ============================================================================
# 9. Binary PLY export (CloudCompare-friendly)
# ============================================================================
def save_foundation_ply(path: str | Path,
                        pts: np.ndarray,
                        scalar_field: np.ndarray,
                        field_name: str = "planarity") -> None:
    """Save a binary little-endian PLY with x, y, z and one float scalar field."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"property float {field_name}\n"
        "end_header\n"
    )

    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        (field_name, '<f4'),
    ])
    data = np.zeros(len(pts), dtype=dtype)
    data['x'] = pts[:, 0]
    data['y'] = pts[:, 1]
    data['z'] = pts[:, 2]
    data[field_name] = scalar_field.astype(np.float32)

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())   # single contiguous binary write


# ============================================================================
# 10. Optional visualization of outputs
# ============================================================================
def visualize_outputs(results: dict, save_dir: str | Path = RESULTS_DIR) -> None:
    """Open one window per pipeline output. Opt-in — headless batch runs skip it.

    Saves a planarity histogram PNG, then opens a separate Open3D window for
    each output in sequence (close each window to advance to the next):
        1/5  Input point cloud (centered)
        2/5  Surface normals (on the sampled cloud)
        3/5  Ball-Pivoting mesh (shaded)
        4/5  Occupancy voxel grid
        5/5  Cloud colored by planarity
    and finally shows the planarity histogram window.
    """
    # Lazy import so headless runs don't require matplotlib at all.
    import matplotlib.pyplot as plt

    save_dir  = Path(save_dir)
    centered  = results["centered"]
    sampled   = results["sampled"]
    normals   = results["normals"]
    planarity = results["planarity"]
    mesh      = results["mesh"]
    voxels    = results["voxel_grid"]

    def _show(geoms, name, **kw):
        print(f"[viz] {name}   (close window to continue)...")
        o3d.visualization.draw_geometries(geoms, window_name=name, **kw)

    # ---- Planarity histogram (saved as PNG, shown at the end) ------------
    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / "planarity_histogram.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(planarity, bins=60, color="#3b6fb6", edgecolor="black", linewidth=0.3)
    ax.set_title("Planarity distribution")
    ax.set_xlabel("planarity   (1 = flat, 0 = linear / scattered)")
    ax.set_ylabel("point count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"[viz] Saved planarity histogram -> {png_path}")

    # ---- 1/5. Input point cloud (centered) -------------------------------
    pcd_in = o3d.geometry.PointCloud()
    pcd_in.points = o3d.utility.Vector3dVector(centered)
    pcd_in.paint_uniform_color([0.6, 0.6, 0.6])
    _show([pcd_in], "1/5  Input cloud (centered)")

    # ---- 2/5. Surface normals (on the sampled cloud) ---------------------
    pcd_n = o3d.geometry.PointCloud()
    pcd_n.points  = o3d.utility.Vector3dVector(sampled)
    pcd_n.normals = o3d.utility.Vector3dVector(normals)
    _show([pcd_n], "2/5  Surface normals", point_show_normal=True)

    # ---- 3/5. Mesh -------------------------------------------------------
    _show([mesh], "3/5  Ball-Pivoting mesh", mesh_show_back_face=True)

    # ---- 4/5. Voxel grid -------------------------------------------------
    _show([voxels], "4/5  Voxel grid")

    # ---- 5/5. Planarity-colored cloud ------------------------------------
    norm   = (planarity - planarity.min()) / (np.ptp(planarity) + 1e-8)
    colors = plt.get_cmap(CMAP_NAME)(norm)[:, :3]
    pcd_p = o3d.geometry.PointCloud()
    pcd_p.points = o3d.utility.Vector3dVector(centered)
    pcd_p.colors = o3d.utility.Vector3dVector(colors)
    _show([pcd_p], "5/5  Planarity (viridis)")

    # ---- Planarity histogram window (blocks until closed) ----------------
    plt.show()
    plt.close(fig)


# ============================================================================
# Pipeline orchestration
# ============================================================================
def run_pipeline(visualize: bool = VISUALIZE) -> dict:
    timings: dict[str, float] = {}

    # ---- 1. Load ----------------------------------------------------------
    t0 = time.time()
    pcd_raw = load_pc(INPUT_PATH)
    pc_raw  = np.asarray(pcd_raw.points)
    timings["load"] = time.time() - t0
    print(f"[1] Loaded {len(pc_raw):,} points")

    # ---- 2. Stats & centering --------------------------------------------
    t0 = time.time()
    p_min, p_max, center = compute_spatial_stats(pc_raw)
    pc_centered = pc_raw - center
    timings["center"] = time.time() - t0
    print(f"[2] Bbox = {p_max - p_min}   center = {center}")

    # ---- 3. Voxel downsample ---------------------------------------------
    t0 = time.time()
    pc_sampled = smart_voxel_sample(pc_centered, VOXEL_SIZE_SAMPLE)
    timings["sample"] = time.time() - t0
    print(f"[3] Sampled: {len(pc_centered):,} -> {len(pc_sampled):,} "
          f"({100 * (1 - len(pc_sampled) / len(pc_centered)):.1f}% reduction)")

    # ---- 4. Normals -------------------------------------------------------
    t0 = time.time()
    pcd_normals = estimate_normals(pc_sampled, k=K_NORMALS)
    normals = np.asarray(pcd_normals.normals)
    timings["normals"] = time.time() - t0
    print(f"[4] Normals computed, shape = {normals.shape}")

    # ---- 5. Mesh reconstruction ------------------------------------------
    t0 = time.time()
    mesh = reconstruct_mesh(pc_sampled)
    timings["mesh"] = time.time() - t0
    print(f"[5] Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # ---- 6. Voxelization --------------------------------------------------
    t0 = time.time()
    pcd_sampled = o3d.geometry.PointCloud()
    pcd_sampled.points = o3d.utility.Vector3dVector(pc_sampled)
    pcd_sampled.paint_uniform_color([0.6, 0.6, 0.6])
    voxel_grid = voxelize_cloud(pcd_sampled, VOXEL_SIZE_GRID)
    n_voxels = len(voxel_grid.get_voxels())
    timings["voxelize"] = time.time() - t0
    print(f"[6] Voxel grid ({VOXEL_SIZE_GRID*100:.0f} cm): {n_voxels:,} occupied cells")

    # ---- 7. Planarity feature --------------------------------------------
    t0 = time.time()
    planarity = compute_planarity(pc_sampled, k=K_PLANARITY)
    timings["planarity"] = time.time() - t0
    print(f"[7] Planarity range = [{planarity.min():.3f}, {planarity.max():.3f}]  "
          f"mean = {planarity.mean():.3f}")

    # ---- 8. Project back to full resolution ------------------------------
    t0 = time.time()
    full_planarity = project_scalars(pc_centered, pc_sampled, planarity)
    timings["project"] = time.time() - t0
    print(f"[8] Projected planarity to {len(pc_centered):,} points")

    # ---- 9. Export --------------------------------------------------------
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ply_out  = RESULTS_DIR / "foundations.ply"
    mesh_out = RESULTS_DIR / "mesh.ply"
    save_foundation_ply(ply_out, pc_raw, full_planarity, field_name="planarity")
    o3d.io.write_triangle_mesh(str(mesh_out), mesh)
    # Also stash the center vector so the export can be georeferenced later.
    np.save(RESULTS_DIR / "center.npy", center)
    timings["export"] = time.time() - t0
    print(f"[9] Wrote {ply_out}")
    print(f"    Wrote {mesh_out}")
    print(f"    Wrote {RESULTS_DIR / 'center.npy'}  (offset for georeferencing)")

    # ---- Summary ----------------------------------------------------------
    print()
    print("=" * 52)
    print("MISSION 01: DATA FOUNDATIONS SUMMARY")
    print("=" * 52)
    print(f"Input points     : {len(pc_raw):,}")
    print(f"Sampled points   : {len(pc_sampled):,}")
    print(f"Mesh triangles   : {len(mesh.triangles):,}")
    print(f"Occupied voxels  : {n_voxels:,}")
    print(f"Planarity range  : [{full_planarity.min():.3f}, {full_planarity.max():.3f}]")
    print("-" * 52)
    for step, duration in timings.items():
        print(f"{step:<15}: {duration:8.4f} s")
    print(f"{'TOTAL':<15}: {sum(timings.values()):8.4f} s")
    print("=" * 52)

    results = {
        "raw":         pc_raw,
        "centered":    pc_centered,
        "sampled":     pc_sampled,
        "center":      center,
        "normals":     normals,
        "mesh":        mesh,
        "voxel_grid":  voxel_grid,
        "planarity":   full_planarity,
        "timings":     timings,
    }

    # ---- 10. Optional visualization --------------------------------------
    if visualize:
        visualize_outputs(results)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mission 01 — Data Foundations pipeline")
    parser.add_argument(
        "--viz", action="store_true",
        help="force-render outputs even if VISUALIZE is False")
    parser.add_argument(
        "--no-viz", action="store_true",
        help="skip visualization (headless: console + files only)")
    # parse_known_args so IDE wrappers (e.g. Spyder's runfile --wdir) don't break parsing.
    args, _ = parser.parse_known_args()

    show_viz = (VISUALIZE or args.viz) and not args.no_viz
    run_pipeline(visualize=show_viz)
