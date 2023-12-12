import random
from file_utils import *
import glob
import shutil

NORMALIZED_DATA_DIR = "../data_iiitd/tmp/normalized_all"
OUTPUT_TEST_VAL_DIR = "../data_iiitd/tmp/normalized_splitted"

#this function divides normalized images in train and val subsets, mandataory for the training phase
def copy_to_train_val(normalized_images_dir: str = NORMALIZED_DATA_DIR,
                      target_dir: str = OUTPUT_TEST_VAL_DIR,
                      test_size: float = 0.2, 
                      random_state: int = 1):
    random.seed(random_state)

    shutil.rmtree(OUTPUT_TEST_VAL_DIR)
    assert 0 < test_size < 1.0
    samples = {}

    normalized_images_paths = sorted(glob.glob(f"{normalized_images_dir}/*"))

    for path in normalized_images_paths:
        user_id, img_type, _ = extract_user_sample_ids(path)
        user_samples = samples.get(user_id, [])
        user_samples.append(path)
        samples[user_id] = user_samples

    train_dir = f"{target_dir}/train"
    create_empty_dir(train_dir)
    val_dir = f"{target_dir}/val"
    create_empty_dir(val_dir)

    for user_id, user_samples in samples.items():
        #We take all the indices belonging to the dataset
        indices = list(range(len(user_samples)))
        random.shuffle(indices)

        #We create the three subsets with probes belonging to the Normal, Colored, Transparent irises
        normal = [user_samples[i] for i in indices if "Normal" in user_samples[i]]
        colored = [user_samples[i] for i in indices if "Colored" in user_samples[i]]
        transparent = [user_samples[i] for i in indices if "Transparent" in user_samples[i]]
        
        #We initialize, for the three lists, the split variable (test_size = 0.2)
        split_normal = int(len(normal) * (1.0 - test_size))
        split_colored = int(len(colored) *(1.0-test_size))
        split_transparent = int(len(colored) *(1.0-test_size))
        
        #We create the train set with the names of the samples of the three lists (splitted with the indices), merged together
        train = [i for i in normal[:split_normal]] + [i for i in colored[:split_colored]] + [i for i in transparent[:split_transparent]]
        #We shuffle the train set 
        random.shuffle(train)
        #We do the same for the validation set 
        val = [i for i in normal[split_normal:]] + [i for i in colored[split_colored:]] + [i for i in transparent[split_transparent:]]
        random.shuffle(val)
        

        train_user_dir = f"{train_dir}/{user_id}"
        create_empty_dir(train_user_dir)
        copy_dataset(train, train_user_dir)

        val_user_dir = f"{val_dir}/{user_id}"
        create_empty_dir(val_user_dir)
        copy_dataset(val, val_user_dir)


if __name__ == '__main__':
    copy_to_train_val()
