import pyvista as pv
import numpy as np
import time


def animate_traj(traj, traj_R,duration_sec=100, fps=60, stl_file="Assembly.STL"):
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
    R_data = traj_R[::step]
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

    # --- Base point sphere (initialize at first node) ---
    base_sphere = pv.Sphere(radius=0.1, center=pts0[0])
    plotter.add_mesh(base_sphere, color="blue", name="base_sphere")
    # --- Segments (every 2 nodes) ---
    # segments = []
    # for i in range(0, len(pts0)-1, 2):  # step=2 leaves gaps
    #     seg = pv.Line(pts0[i], pts0[i+1])
    #     segments.append(seg)
    # dashed_line = pv.MultiBlock(segments).combine()
    # seg_actor = plotter.add_mesh(dashed_line, color="blue", line_width=3)
    # --- Grid ---
    grid_size, grid_res = 1000, 100
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
    # plotter.show(interactive_update=True)
    out_name = "mecc1.5.mp4"
    plotter.open_movie(out_name, framerate=fps)
    # --- Animate ---
    for i in range(T):
        pts = data[i].reshape(N_plus_1, 3)
        polyline.points = pts
        R = R_data[i]
        # move drone mesh to tip
        new_point = pts[-1]
        displacement = new_point - start_point
        drone_actor.points = stl_mesh.points @ R.T + new_point#stl_mesh.points + displacement + start_point

        # move base sphere to first node
        base_sphere.points = pv.Sphere(radius=0.1, center=pts[0]).points

        text_actor.SetText(0, f"t = {i*dt:.3f} s")
        plotter.camera_position = [
        (cam_pos[0]*np.sin(0.1*i*dt), cam_pos[1]*np.cos(0.1*i*dt), cam_pos[2]),  # camera pos
        (cam_pos[0], 0, cam_pos[2]),           # focal point
        (0, 0, 1),
    ]
        plotter.update()
        plotter.write_frame()
        time.sleep(dt)

    plotter.close()

import pyvista as pv
import numpy as np
import time


def animate_trajc(traj,duration_sec=100, fps=60, stl_file="Assembly.STL"):
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

    # --- Base point sphere (initialize at first node) ---
    base_sphere = pv.Sphere(radius=0.1, center=pts0[0])
    plotter.add_mesh(base_sphere, color="blue", name="base_sphere")
    # --- Segments (every 2 nodes) ---
    # segments = []
    # for i in range(0, len(pts0)-1, 2):  # step=2 leaves gaps
    #     seg = pv.Line(pts0[i], pts0[i+1])
    #     segments.append(seg)
    # dashed_line = pv.MultiBlock(segments).combine()
    # seg_actor = plotter.add_mesh(dashed_line, color="blue", line_width=3)
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
    cam_pos = pts0[0] + np.array([0, -120, 50])
    plotter.camera_position = [
        (cam_pos[0], cam_pos[1], cam_pos[2]),  # camera pos
        (cam_pos[0], 0, cam_pos[2]),           # focal point
        (0, 0, 1),
    ]

    plotter.show_axes()
    plotter.enable_anti_aliasing("fxaa")
    base_traj = [pts0[0]]
    base_line = pv.PolyData(np.array(base_traj))
    plotter.add_mesh(base_line, color="green", line_width=3, name="base_traj")

    # --- Interactive window ---
    plotter.show(interactive_update=True)
    out_name = "mecc1.5.mp4"
    plotter.open_movie(out_name, framerate=fps)
    # --- Animate ---
    for i in range(T):
        pts = data[i].reshape(N_plus_1, 3)
        polyline.points = pts
        # --- Update base trajectory (green line) ---
        base_traj.append(pts[0])
        line_pts = np.array(base_traj)
        npts = len(line_pts)
        lines = np.hstack([[npts, *range(npts)]])
        new_line = pv.PolyData(line_pts, lines=lines)

        plotter.remove_actor("base_traj", reset_camera=False)
        plotter.add_mesh(new_line, color="green", line_width=3, name="base_traj")

        # move drone mesh to tip
        new_point = pts[-1]
        displacement = new_point - start_point
        drone_actor.points = stl_mesh.points + new_point#stl_mesh.points + displacement + start_point
        plotter.camera_position = [
        (cam_pos[1]*np.sin(0.05*i*dt), cam_pos[1]*np.cos(0.05*i*dt), cam_pos[2]),  # camera pos
        (cam_pos[0], 0, cam_pos[2]-50),           # focal point
        (0, 0, 1),
        ]

        # move base sphere to first node
        base_sphere.points = pv.Sphere(radius=0.1, center=pts[0]).points

        text_actor.SetText(0, f"t = {i*dt:.3f} s")

        plotter.update()
        plotter.write_frame()
        time.sleep(dt)

    plotter.close()

