import glob
from PIL import Image

def write_to_gif(dir = "temp_figures"):
    """Create GIF from saved figures with extended first and last frame durations."""
    print("Creating GIF from training progress figures...")
    try:
        # Get all PNG files in temp_figures directory
        image_files = sorted(glob.glob(f"{dir}/*.png"))
        
        if image_files:
            # Open all images
            images = []
            for filename in image_files:
                img = Image.open(filename)
                images.append(img)
            
            # Create duration list: first and last frames stay longer
            durations = []
            for i in range(len(images)):
                if i == 0:  # First frame
                    durations.append(2000)  # 2 seconds
                elif i == len(images) - 1:  # Last frame
                    durations.append(3000)  # 3 seconds
                else:  # Middle frames
                    durations.append(500)   # 0.5 seconds
            
            # Save as GIF in main directory
            gif_filename = "animation.gif"
            images[0].save(
                gif_filename,
                save_all=True,
                append_images=images[1:],
                duration=durations,
                loop=0
            )
            print(f"GIF created successfully: {gif_filename}")
            print(f"Total frames: {len(images)}")
            print(f"First frame duration: {durations[0]}ms, Last frame duration: {durations[-1]}ms")
            
        else:
            print("No PNG files found to create GIF")
            
    except Exception as e:
        print(f"Error creating GIF: {e}")