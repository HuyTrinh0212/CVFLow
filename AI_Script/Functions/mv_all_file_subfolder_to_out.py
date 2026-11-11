import os
import shutil

def flatten_images(image_folder):
    file_type = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.txt', '.json', '.yaml', '.xml'}

    for root, dirs, files in os.walk(image_folder, topdown=False):
        if root == image_folder:
            continue

        for file in files:
            if os.path.splitext(file)[1].lower() in file_type:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(image_folder, file)

                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(file)
                    i = 1
                    while os.path.exists(os.path.join(image_folder, f"{base}_{i}{ext}")):
                        i += 1
                    dst_path = os.path.join(image_folder, f"{base}_{i}{ext}")

                shutil.move(src_path, dst_path)
                print(f"Moved: {src_path} -> {dst_path}")

        if not os.listdir(root):
            os.rmdir(root)
            print(f"Removed empty folder: {root}")


if __name__ == "__main__":
    image_folder = "/home/bht/CODE/AI_Script/inputs/dataset/Dataset_Number_New/labels/"
    flatten_images(image_folder)