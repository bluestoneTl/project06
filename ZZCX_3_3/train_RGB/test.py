import os

def find_images_with_ce1(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) and 'ce1' in file:
                image_files.append(os.path.join(root, file))
    return image_files

def rename_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) and 'ce1' in file:
                new_name = file
                parts = file.split('_')
                new_parts = []
                flag = False
                for part in parts:
                    if part == '3y' and not flag:
                        new_parts.append(part)
                        flag = True
                    elif part != '3y':
                        new_parts.append(part)
                new_name = '_'.join(new_parts)
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_name)
                if new_name != file:
                    os.rename(old_path, new_path)
                
if __name__ == '__main__':
    folder_path = 'datasets/ZZCX_2_1/train_RGB/HQ'  # 当前文件夹，可修改为目标文件夹路径
    rename_images(folder_path)
    result = find_images_with_ce1(folder_path)
    for image in result:
        print(image)