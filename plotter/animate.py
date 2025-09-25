import pyvista as pv
import numpy as np
import time


def animate_traj(traj, duration_sec=100, fps=60, stl_file="Assembly.STL"):
    """
    Animate a trajectory of string + drone using PyVista.

    Args:
        traj : ndarray of shape (num_steps, flat_dim) or (num_steps, N+1, 3)
        duration_sec (float): target animation duration in seconds
        fps (int): frames per second
        stl_file (str): path to drone STL file
    """
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

    # --- Downsample for desired fps & duration ---
    step = int(max(1, num_steps // (duration_sec * fps)))
    data = traj[::step]
    T = data.shape[0]
    dt = duration_sec / T

    # --- Initial polyline ---
    pts0 = data[0].reshape(N_plus_1, 3)
    lines = np.hstack([[N_plus_1, *range(N_plus_1)]])
    polyline = pv.PolyData(pts0, lines=lines)

    plotter = pv.Plotter()
    plotter.add_mesh(polyline, color="red", line_width=4)
    text_actor = plotter.add_text("t=0.000 s", position="upper_left",
                                  font_size=14, color="black")

    # --- Grid ---
    grid_size, grid_res = 100, 50
    grid_lines = []
    for x in np.linspace(-grid_size/2, grid_size/2, grid_res):
        grid_lines.append(pv.Line((x, -grid_size/2, 0), (x, grid_size/2, 0)))
    for y in np.linspace(-grid_size/2, grid_size/2, grid_res):
        grid_lines.append(pv.Line((-grid_size/2, y, 0), (grid_size/2, y, 0)))
    grid = pv.MultiBlock(grid_lines).combine()
    plotter.add_mesh(grid, color="black", line_width=1)

    # --- STL object (drone) ---
    stl_mesh = pv.read(stl_file)
    stl_mesh.scale([5, 5, 5], inplace=True)
    stl_mesh.rotate_x(90, inplace=True)

    start_point = pts0[-1]
    drone_actor = stl_mesh.copy()
    drone_actor.points += start_point
    plotter.add_mesh(drone_actor, name="stl_actor", color="lightblue")

    # --- Camera ---
    cam_pos = pts0[0] + np.array([0, -40, 10])
    plotter.camera_position = [
        (cam_pos[0], cam_pos[1], cam_pos[2]),  # camera pos
        (cam_pos[0], 0, cam_pos[2]),           # focal point
        (0, 0, 1),
    ]

    plotter.show_axes()
    plotter.enable_anti_aliasing("fxaa")

    # --- Interactive window ---
    plotter.show(interactive_update=True)

    # --- Animate ---
    for i in range(T):
        pts = data[i].reshape(N_plus_1, 3)
        polyline.points = pts

        # move drone mesh to tip
        new_point = pts[-1]
        displacement = new_point - start_point
        drone_actor.points = stl_mesh.points + displacement + start_point

        text_actor.SetText(0, f"t = {i*dt:.3f} s")

        plotter.update()
        time.sleep(dt)

    plotter.close()
