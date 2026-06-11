import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_traj_frames_pdf(traj, frame_ids=None, pdf_name="traj_frames.pdf", stl_file="Assembly.STL"):
    # --- Reshape trajectory if flat ---
    if traj.ndim == 2:
        num_steps, flat_dim = traj.shape
        N_plus_1 = flat_dim // 3
    elif traj.ndim == 3:
        num_steps, N_plus_1, _ = traj.shape
        flat_dim = N_plus_1 * 3
        traj = traj.reshape(num_steps, flat_dim)
    else:
        raise ValueError("traj must have shape (T, flat_dim) or (T, N+1, 3)")

    if frame_ids is None:
        # pick 10 evenly spaced frames by default
        frame_ids = np.linspace(0, num_steps - 1, 10, dtype=int)
    else:
        frame_ids = np.asarray(frame_ids, dtype=int)

    # --- Shared static geometry ---
    pts0 = traj[0].reshape(N_plus_1, 3)
    lines = np.hstack([[N_plus_1, *range(N_plus_1)]])

    # grid
    grid_size, grid_res = 100, 50
    grid_lines = []
    for x in np.linspace(-grid_size / 2, grid_size / 2, grid_res):
        grid_lines.append(pv.Line((x, -grid_size / 2, 0), (x, grid_size / 2, 0)))
    for y in np.linspace(-grid_size / 2, grid_size / 2, grid_res):
        grid_lines.append(pv.Line((-grid_size / 2, y, 0), (grid_size / 2, y, 0)))
    grid = pv.MultiBlock(grid_lines).combine()

    # STL object
    stl_mesh = pv.read(stl_file)
    stl_mesh.scale([5, 5, 5], inplace=True)
    stl_mesh.rotate_x(90, inplace=True)

    with PdfPages(pdf_name) as pdf:
        base_traj = []

        for idx in frame_ids:
            pts = traj[idx].reshape(N_plus_1, 3)

            plotter = pv.Plotter(off_screen=True, window_size=(1400, 1000))
            plotter.add_mesh(grid, color="black", line_width=1)

            # trajectory polyline
            polyline = pv.PolyData(pts, lines=lines)
            plotter.add_mesh(polyline, color="red", line_width=4)

            # base sphere
            base_sphere = pv.Sphere(radius=0.1, center=pts[0])
            plotter.add_mesh(base_sphere, color="blue")

            # base trajectory up to this frame
            base_traj.append(pts[0])
            if len(base_traj) >= 2:
                line_pts = np.array(base_traj)
                npts = len(line_pts)
                traj_lines = np.hstack([[npts, *range(npts)]])
                base_line = pv.PolyData(line_pts, lines=traj_lines)
                plotter.add_mesh(base_line, color="green", line_width=3)

            # drone at tip
            drone_actor = stl_mesh.copy()
            drone_actor.points = stl_mesh.points + pts[-1]
            plotter.add_mesh(drone_actor, color="lightblue")

            # camera
            cam_pos = pts[0] + np.array([0, -25, 5])
            plotter.camera_position = [
                (cam_pos[0], cam_pos[1], cam_pos[2]),
                (cam_pos[0], 0, cam_pos[2]),
                (0, 0, 1),
            ]

            plotter.show_axes()
            img = plotter.screenshot(return_img=True)
            plotter.close()

            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Frame {idx} / {num_steps - 1}")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved PDF to {pdf_name}")

from plotter.animate import animate_trajc
import numpy as np
import matplotlib.pyplot as plt
# === Test file ===
test_file = "results/sim_rank90_omega0.900.npz"

# === Simulation parameters ===



h = 1e-4




# # === Load test data ===
data = np.load(test_file)
traj_nodes = data["trajectories"]  # (T, N, 3)
print(traj_nodes.shape)
num_steps, num_nodes = traj_nodes.shape

time = np.arange(num_steps) * h
spk_val = traj_nodes[:,0]
spk_dot = -(spk_val[1:] - spk_val[:-1])/h
traj_nodes = traj_nodes[:,1:]

num_steps = traj_nodes.shape[0]
num_nodes = traj_nodes.shape[1] // 3  # 6 // 3 = 2

traj_nodes = traj_nodes.reshape(num_steps, num_nodes, 3)

save_traj_frames_pdf(
    np.array(traj_nodes),
    frame_ids=[0, 50/h, 100/h, 150/h, 190/h],
    pdf_name="selected_frames.pdf",
    stl_file="models/Assembly.STL"
)