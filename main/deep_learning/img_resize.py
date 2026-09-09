import os
import sys
from PIL import Image

def resize_images(input_folder, output_folder, width, height):

    os.makedirs(output_folder, exist_ok=True)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    count = 0
    for filename in os.listdir(input_folder):

        if filename.lower().endswith(valid_ext):

            input_path = os.path.join(input_folder, filename)

            name, ext = os.path.splitext(filename)

            new_name = f"{name}_{width}x{height}{ext}"
            output_path = os.path.join(output_folder, new_name)

            try:
                img = Image.open(input_path)
                img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
                img_resized.save(output_path)
                count += 1
            except Exception as e:
                print(f"Error to process {filename}: {e}")

    print(f"{count} Resized images saved at: {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Use: python img_resize.py <input_folder_path> <output_folder_path> <length> <width>")
    else:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        width = int(sys.argv[3])
        height = int(sys.argv[4])
        resize_images(input_folder, output_folder, width, height)
