"""将注释数据从xml格式转换为txt格式
"""

import os
import xml.etree.ElementTree as ET


xml_dir = r"C:\Users\he81t\ubuntu\yamaguchi\fukuoka_chicken\original_data\Annotations_pascal_xml"  # XML文件所在目录
txt_dir = r"C:\Users\he81t\ubuntu\yamaguchi\fukuoka_chicken\original_data\Annotations_yolo_txt"    # 输出YOLO格式的txt文件的目录

if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)

# 存储类别列表
classes = []

# 遍历xml_dir中的文件
for filename in os.listdir(xml_dir):
    # 仅处理扩展名为.xml的文件
    if filename.endswith(".xml"):
        # XML文件路径
        xml_path = os.path.join(xml_dir, filename)

        # 解析XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find("size")
        if size is None:
            # 如果没有size标签则跳过
            continue

        width = float(size.find("width").text)
        height = float(size.find("height").text)

        # 输出文本文件路径（替换为.txt）
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(txt_dir, txt_filename)

        # 用于写入YOLO格式注释的列表
        yolo_annotations = []

        # 获取所有object标签并转换为YOLO格式
        for obj in root.findall("object"):
            # 类别名称
            class_name = obj.find("name").text
            if class_name not in classes:
                classes.append(class_name)
            class_id = classes.index(class_name)

            # 边界框坐标
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # YOLO格式为 (class_id, x_center, y_center, w, h) [已归一化]
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            # 如果需要调整小数点后的位数以便写出，可以在此处使用format
            annotation_str = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            yolo_annotations.append(annotation_str)

        # 写入文本文件
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in yolo_annotations:
                f.write(line + "\n")

# 转换完成后，如果需要检查存储在classes中的类别列表，可以单独输出。
print("类别列表:", classes)