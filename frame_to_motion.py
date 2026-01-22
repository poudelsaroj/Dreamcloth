import open3d as o3d
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Configuration
mesh_dir = Path("./smpl_meshes_from_bvh")
output_dir = Path("./rendered_frames")
output_video = "animation.mp4"
fps = 20  # Match your BVH fps (50ms frame time = 20 fps)
num_frames = 188

# Create output directory
output_dir.mkdir(exist_ok=True)

print("Loading first mesh to determine camera settings...")
first_mesh = o3d.io.read_triangle_mesh(str(mesh_dir / "frame_00000.obj"))
first_mesh.compute_vertex_normals()

# Get mesh statistics
vertices = np.asarray(first_mesh.vertices)
print(f"Mesh has {len(vertices)} vertices")
print(f"Vertex range: X[{vertices[:,0].min():.2f}, {vertices[:,0].max():.2f}], "
      f"Y[{vertices[:,1].min():.2f}, {vertices[:,1].max():.2f}], "
      f"Z[{vertices[:,2].min():.2f}, {vertices[:,2].max():.2f}]")

# Get bounds
bounds = first_mesh.get_axis_aligned_bounding_box()
center = bounds.get_center()
extent = bounds.get_extent()
max_extent = max(extent)

print(f"Mesh center: {center}")
print(f"Mesh extent: {extent}")
print(f"Max extent: {max_extent:.3f}")

# Create visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=800, visible=True)  # Set visible=True to debug

# Render options
opt = vis.get_render_option()
opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray background
opt.mesh_show_back_face = True
opt.light_on = True
opt.point_size = 1.0

# Add first mesh with color
first_mesh.paint_uniform_color([0.8, 0.6, 0.4])  # Skin-like color
vis.add_geometry(first_mesh)

# Set up camera
ctr = vis.get_view_control()

# Method 1: Auto-reset view (usually works)
ctr.set_zoom(0.7)

# Method 2: Manual camera setup (if auto doesn't work)
# Calculate camera position
camera_distance = max_extent * 2.5
camera_pos = center + np.array([0, 0, camera_distance])

# These might not work in older Open3D versions, try uncommenting:
# ctr.set_lookat(center)
# ctr.set_front([0, 0, -1])
# ctr.set_up([0, 1, 0])

print(f"Camera distance: {camera_distance:.3f}")

# Test render first frame
vis.poll_events()
vis.update_renderer()

# Capture test frame
print("Capturing test frame...")
image = vis.capture_screen_float_buffer(do_render=True)
test_img = (np.asarray(image) * 255).astype(np.uint8)

# Check if test frame is black
if test_img.max() < 10:
    print("⚠️ WARNING: Test frame is black!")
    print("The mesh might not be visible. Try:")
    print("1. Check if mesh loaded correctly")
    print("2. Adjust camera zoom/position")
    print("3. View mesh manually: o3d.visualization.draw_geometries([mesh])")
else:
    print(f"✅ Test frame looks good (max pixel value: {test_img.max()})")

# Save test frame
cv2.imwrite("test_frame.png", cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
print("Saved test_frame.png - check if mesh is visible!")

# Ask user to confirm before proceeding
response = input("\nDoes test_frame.png look correct? (y/n): ")
if response.lower() != 'y':
    print("Exiting. Adjust camera settings and try again.")
    vis.destroy_window()
    exit()

# Render all frames
print("\nRendering all frames...")
for i in tqdm(range(num_frames), desc="Rendering"):
    mesh_path = mesh_dir / f"frame_{i:05d}.obj"
    
    if not mesh_path.exists():
        print(f"Warning: {mesh_path} not found, skipping")
        continue
    
    # Clear and load new mesh
    vis.clear_geometries()
    
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.6, 0.4])  # Skin tone
    
    vis.add_geometry(mesh, reset_bounding_box=False)  # Don't reset camera
    vis.poll_events()
    vis.update_renderer()
    
    # Capture
    image = vis.capture_screen_float_buffer(do_render=True)
    img_np = (np.asarray(image) * 255).astype(np.uint8)
    
    # Save
    out_path = output_dir / f"frame_{i:05d}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

vis.destroy_window()
print(f"\n✅ Rendered {num_frames} frames to '{output_dir}/'")

# Create video
print("Creating video with ffmpeg...")
import subprocess

try:
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', f'{output_dir}/frame_%05d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_video
    ], check=True)
    print(f"✅ Video saved as '{output_video}'")
except subprocess.CalledProcessError:
    print("ffmpeg failed, trying OpenCV...")
    
    # Fallback to OpenCV
    frame_files = sorted(output_dir.glob("frame_*.png"))
    first_frame = cv2.imread(str(frame_files[0]))
    h, w = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    for f in tqdm(frame_files, desc="Writing video"):
        frame = cv2.imread(str(f))
        writer.write(frame)
    
    writer.release()
    print(f"✅ Video saved as '{output_video}'")