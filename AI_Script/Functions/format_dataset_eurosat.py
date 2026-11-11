# import os
#
# # Mapping class name sang index (dựa trên thông tin bạn cung cấp)
# class_mapping = {
#     "Forest": "0",
#     "River": "1",
#     "Highway": "2",
#     "AnnualCrop": "3",
#     "SeaLake": "4",
#     "HerbaceousVegetation": "5",
#     "Industrial": "6",
#     "Residential": "7",
#     "PermanentCrop": "8",
#     "Pasture": "9"
# }
#
# # Đường dẫn thư mục gốc chứa dataset
# dataset_dir = "/home/bht/AI/data/EuroSAT"
# # Thư mục đầu ra cho các file label
# output_dir = "/home/bht/AI/data/EuroSAT/labels"
#
# # Tạo thư mục labels nếu chưa tồn tại
# os.makedirs(output_dir, exist_ok=True)
#
# # Phần mở rộng ảnh được hỗ trợ
# image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
#
# # Kiểm tra trùng tên file để tránh ghi đè
# existing_labels = set()
#
# print("Đang xử lý dataset...")
#
# # Duyệt qua tất cả các thư mục con trong dataset
# for class_name in os.listdir(dataset_dir):
#     class_path = os.path.join(dataset_dir, class_name)
#
#     # Bỏ qua nếu không phải thư mục
#     if not os.path.isdir(class_path):
#         continue
#
#     # Kiểm tra class có trong mapping không
#     if class_name not in class_mapping:
#         print(f"⚠️ Cảnh báo: Thư mục '{class_name}' không có trong mapping. Bỏ qua!")
#         continue
#
#     class_label = class_mapping[class_name]
#     print(f"\nXử lý class: {class_name} (label = {class_label})")
#
#     # Duyệt qua tất cả file trong thư mục class
#     for img_file in os.listdir(class_path):
#         # Lấy phần mở rộng và kiểm tra có phải file ảnh
#         _, ext = os.path.splitext(img_file)
#         if ext.lower() not in image_extensions:
#             continue
#
#         # Tạo tên file label tương ứng
#         base_name = os.path.splitext(img_file)[0]
#         label_filename = f"{base_name}.txt"
#         label_path = os.path.join(output_dir, label_filename)
#
#         # Kiểm tra trùng tên file
#         if base_name in existing_labels:
#             print(f"❗️ Cảnh báo: Trùng tên file '{img_file}'. File label sẽ bị ghi đè!")
#         else:
#             existing_labels.add(base_name)
#
#         # Ghi file label
#         with open(label_path, 'w') as f:
#             f.write(class_label)
#
#         print(f"  ✅ Tạo {label_filename} cho {img_file}")
#
# print("\nHoàn tất!")
# print(f"Tổng số file label đã tạo: {len(existing_labels)}")
# print(f"Kết quả nằm trong thư mục: {os.path.abspath(output_dir)}")


# import os
# import shutil
#
# # Cấu hình đường dẫn
# source_dir = "/home/bht/AI/data/EuroSAT"
# dest_dir = "/home/bht/AI/data/EuroSAT_Formatted/images"
#
# # Các định dạng ảnh được hỗ trợ (phân biệt chữ thường/chữ hoa)
# image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
#
# # Tạo thư mục đích nếu chưa tồn tại
# os.makedirs(dest_dir, exist_ok=True)
#
# # Đếm số file đã xử lý
# moved_files = 0
# duplicate_files = 0
#
# print(f"🔍 Bắt đầu quét và di chuyển ảnh từ '{source_dir}' sang '{dest_dir}'...")
#
# # Duyệt tất cả các thư mục con và file trong source_dir
# for root, _, files in os.walk(source_dir):
#     for file in files:
#         # Lấy phần mở rộng và chuẩn hóa về chữ thường
#         _, ext = os.path.splitext(file)
#         if ext.lower() not in image_extensions:
#             continue  # Bỏ qua nếu không phải định dạng ảnh hợp lệ
#
#         # Đường dẫn nguồn và đích
#         src_path = os.path.join(root, file)
#         dest_path = os.path.join(dest_dir, file)
#
#         # Kiểm tra file trùng tên
#         if os.path.exists(dest_path):
#             duplicate_files += 1
#             print(f"⚠️  [TRÙNG LẶP] File '{file}' đã tồn tại trong '{dest_dir}'. Ghi đè!")
#
#         # Di chuyển file
#         shutil.move(src_path, dest_path)
#         moved_files += 1
#         print(f"✅ Đã di chuyển: {file} -> {os.path.relpath(dest_path)}")
#
# # Thông báo kết quả
# print("\n🎉 HOÀN TẤT!")
# print(f"✅ Tổng số file đã di chuyển: {moved_files}")
# if duplicate_files > 0:
#     print(f"❗ Số file bị ghi đè do trùng tên: {duplicate_files}")
# print(f"📁 Đường dẫn thư mục ảnh: {os.path.abspath(dest_dir)}")
#
# # Lưu ý quan trọng
# print("\nℹ️  Lưu ý:")
# print("- Các thư mục con trong 'Dataset' sẽ trở nên trống sau khi di chuyển")
# print("- File trùng tên sẽ được ghi đè (file mới nhất sẽ được giữ lại)")
# print("- Kiến trúc thư mục gốc sẽ bị thay đổi vĩnh viễn")



import os
import re

folder = "/home/bht/AI/data/EuroSAT_Formatted_1000Images/images"

# Regex: find pattern ending with _<number>
pattern = re.compile(r"_(\d+)$|_(\d+)\.")

for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)

    if not os.path.isfile(filepath):
        continue

    # Remove extension first for safer matching
    name_without_ext = os.path.splitext(filename)[0]

    match = pattern.search(name_without_ext)
    if not match:
        continue

    # Extract numeric index
    number = int(match.group(1) or match.group(2))

    # Delete if number > 1000
    if number > 100:
        print("Deleting:", filename)
        os.remove(filepath)
