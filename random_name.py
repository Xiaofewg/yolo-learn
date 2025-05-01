import os
import random
import shutil


def shuffle_yolo_dataset(dataset_dir, output_dir):
    # 获取图像和标注文件列表
    image_files = []
    label_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                image_files.append(os.path.join(root, file))
            elif file.endswith('.txt'):
                label_files.append(os.path.join(root, file))

    # 确保图像和标注文件数量一致
    assert len(image_files) == len(label_files), "图像和标注文件数量不匹配！"

    # 对图像和标注文件进行排序，以确保它们一一对应
    image_files.sort()
    label_files.sort()

    # 生成随机索引
    indices = list(range(len(image_files)))
    random.shuffle(indices)

    # 为每个文件生成新的随机文件名
    for i, index in enumerate(indices):
        image_file = image_files[index]
        label_file = label_files[index]
        image_ext = os.path.splitext(image_file)[1]
        label_ext = os.path.splitext(label_file)[1]
        new_image_name = f"img{i + 1}{image_ext}"
        new_label_name = f"img{i + 1}{label_ext}"
        new_image_path = os.path.join(output_dir, new_image_name)
        new_label_path = os.path.join(output_dir, new_label_name)

        # 复制文件到新路径
        shutil.copy2(image_file, new_image_path)
        shutil.copy2(label_file, new_label_path)


if __name__ == "__main__":
    dataset_dir = "E:\yolov5-My\dataset3"  # 替换为你的数据集目录
    output_dir = "E:\yolov5-My\dataset\images"  # 替换为你指定的输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shuffle_yolo_dataset(dataset_dir, output_dir)
    print("数据集已成功打乱顺序并保存到指定路径。")
    