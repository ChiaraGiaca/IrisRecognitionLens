import os
import re
from glob import glob
from shutil import copyfile
from typing import List, Tuple

FILENAME_REGEX = r"(\d+)_([A-Za-z]+)_(\d+)\.bmp"


def get_file_name(file_path: str) -> str:
    return file_path.split("/")[-1]


def extract_user_sample_ids(image_path: str):
    """
    Extract user ID, image type and sample number from filename
    """
    image_filename = get_file_name(image_path)
    user_id, type_img, sample_id = "", "", ""
    try:
      user_id, type_img, sample_id = re.match(FILENAME_REGEX, image_filename).groups()
    except: 
      print(image_filename)
    return user_id, type_img, sample_id


def create_empty_dir(target_dir: str):
    """
    Create an empty directory, removing all files first if the target directory
    already exists.
    """
    if os.path.exists(target_dir):
        files = glob(target_dir + "/*")
        for file in files:
            os.remove(file)
    else:
        os.makedirs(target_dir)


def copy_dataset(file_paths: List[str], target_dir: str):
    for file_path in file_paths:
        file_name = get_file_name(file_path)
        user_id, type_img, sample_id = extract_user_sample_ids(file_name)
        n = int(len(file_paths)/2)
        if file_path in file_paths[:n]:
          doubled_file = user_id + "_" + type_img + "_" + sample_id + "1" + ".bmp"
          copyfile(file_path, f"{target_dir}/{doubled_file}")
        copyfile(file_path, f"{target_dir}/{file_name}")
        
        
