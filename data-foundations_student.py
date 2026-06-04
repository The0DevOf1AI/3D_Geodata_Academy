# -*- coding: utf-8 -*-
"""
Mission 01: Data Foundations
--------------------------------------------------
* Created by: 🦊 Dr. Florent Poux.
* Copyright: Florent Poux - 3D Geodata Academy.
* License: (c) 3D Geodata Academy
--------------------------------------------------
"""

#%% 1. Imports
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time
from pathlib import Path
from scipy.spatial import cKDTree

# Collects per-step runtimes for the summary printed at the end.
# (Run the whole file top-to-bottom — e.g. Spyder's runfile — so it accumulates.)
timings = {}

#%% 2. Data Loading & Centering

def load_pc(path):
    """Load point cloud from file."""
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points)


path_data = Path("../DATA/indoor_room_labeled.ply")
_t = time.time()
if path_data.exists():
    pc_raw = load_pc(path_data)
else:
    pc_raw = np.random.rand(20000, 3) * 10
timings["load"] = time.time() - _t

# Center the point cloud to avoid high-magnitude coordinate issues.
# Subtract the mean position from all points (lossless: add `center` back to
# recover the originals; keep it for georeferencing exports).
_t = time.time()
center = pc_raw.mean(axis=0)
pc_centered = pc_raw - center
timings["center"] = time.time() - _t

#%% 3. Pre-processing (Smart Voxel Sampling)

def voxel_sample(pts, voxel_size=0.05):
    """Downsample point cloud using a voxel grid."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_down.points)


_t = time.time()
pc_sampled = voxel_sample(pc_centered, 0.05)
timings["sample"] = time.time() - _t

#%% 4. Normal Estimation

def estimate_normals(pts, k=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )

    pcd.orient_normals_consistent_tangent_plane(10)
    return pcd


_t = time.time()
pcd_sampled = estimate_normals(pc_sampled)
timings["normals"] = time.time() - _t

#%% 5. Mesh Reconstruction (Ball Pivoting)

def reconstruct_mesh(pcd):
    """Reconstruct surface from points."""
    # Nearest-neighbor distances reveal the scene's scale so the ball radii can
    # adapt to point density instead of being hardcoded.
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    # Define the radii for the ball pivoting algorithm as multiples of avg_dist:
    # a small ball catches fine detail, a larger ball bridges gaps.
    radii = [avg_dist, avg_dist * 2]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    return mesh


_t = time.time()
mesh_recon = reconstruct_mesh(pcd_sampled)
timings["mesh"] = time.time() - _t

#%% 6. Voxelization

def voxelize(pcd, voxel_size=0.1):
    """Convert to VoxelGrid."""
    v_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    return v_grid


_t = time.time()
voxel_grid = voxelize(pcd_sampled, 0.1)
timings["voxelize"] = time.time() - _t

#%% 7. Geometric Feature Computation (Planarity)

def compute_planarity(pts, k=15):
    """Extract local planarity using PCA."""
    n = len(pts)
    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=k)

    planarity = np.zeros(n)
    for i in range(n):
        neighbors = pts[indices[i]]

        # cov is always 3x3: np.cov treats each of the 3 coordinate rows
        # (neighbors.T) as a variable, regardless of the neighbor count k.
        cov = np.cov(neighbors.T)

        # Eigenvalues sorted ascending: l[0] <= l[1] <= l[2].
        # Planarity = (l1 - l0) / l2, with 1e-8 guarding against divide-by-zero.
        l = np.linalg.eigvalsh(cov)
        planarity[i] = (l[1] - l[0]) / (l[2] + 1e-8)

    return planarity


_t = time.time()
planarity_features = compute_planarity(pc_sampled, k=15)
timings["planarity"] = time.time() - _t

#%% 8. Full-Scale Projection

def project_to_full(pts_full, pts_sampled, labels_sampled):
    """Project results back to original resolution."""
    # Every full-resolution point inherits the feature of its nearest sampled
    # neighbor (workers=-1 runs the query in parallel across all CPU cores).
    tree = cKDTree(pts_sampled)
    _, idx = tree.query(pts_full, k=1, workers=-1)
    return labels_sampled[idx]


_t = time.time()
full_planarity = project_to_full(pc_centered, pc_sampled, planarity_features)
timings["project"] = time.time() - _t

#%% 9. Summary
# Printed before the visualization below, because draw_geometries() blocks
# until you close its window.

print("=" * 52)
print("MISSION 01: DATA FOUNDATIONS SUMMARY")
print("=" * 52)
print(f"Input points     : {len(pc_raw):,}")
print(f"Sampled points   : {len(pc_sampled):,}")
print(f"Mesh triangles   : {len(mesh_recon.triangles):,}")
print(f"Occupied voxels  : {len(voxel_grid.get_voxels()):,}")
print(f"Planarity range  : [{full_planarity.min():.3f}, {full_planarity.max():.3f}]")
print("-" * 52)
for step, duration in timings.items():
    print(f"{step:<15}: {duration:8.4f} s")
print(f"{'TOTAL':<15}: {sum(timings.values()):8.4f} s")
print("=" * 52)

#%% 10. Visual Finalization

def viz_3d(points, scalars=None, title="Final View"):
    """Display points in 3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if scalars is not None:
        norm = (scalars - scalars.min()) / (scalars.max() - scalars.min() + 1e-8)
        colors = plt.cm.viridis(norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=title)


viz_3d(pc_centered, scalars=full_planarity, title="Mission Complete: High-Fidelity Features")
