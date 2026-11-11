import os

def sync_folders(image_folder, label_folder):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_folder)
                   if os.path.splitext(f)[1].lower() in image_extensions}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(label_folder)
                   if os.path.splitext(f)[1].lower() == '.txt'}

    images_to_remove = image_files - label_files
    labels_to_remove = label_files - image_files

    for img in images_to_remove:
        for ext in image_extensions:
            img_path = os.path.join(image_folder, f"{img}{ext}")
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Removed: {img_path}")

    for lbl in labels_to_remove:
        lbl_path = os.path.join(label_folder, f"{lbl}.txt")
        if os.path.exists(lbl_path):
            os.remove(lbl_path)
            print(f"Removed: {lbl_path}")


if __name__ == "__main__":
    image_folder = "/home/bht/AI/data/COCO_Val_2017_1000Images/images"
    label_folder = "/home/bht/AI/data/COCO_Val_2017_1000Images/labels"
    sync_folders(image_folder, label_folder)