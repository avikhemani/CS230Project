import os
import shutil
import random

# source_dir = './AudioMNIST/data/'
# target_dir = './AudioMNIST/data/combined'
#
# for i in range(1, 61):
#     curr_source_dir = source_dir + (str(i) if i >= 10 else "0" + str(i))
#     file_names = os.listdir(curr_source_dir)
#     for file_name in file_names:
#         shutil.move(os.path.join(curr_source_dir, file_name), target_dir)

source_dir = './AudioMNIST/data/combined'
dev_dir = './AudioMNIST/data/dev'
test_dir = './AudioMNIST/data/test'

combined_data = os.listdir(source_dir)
random.shuffle(combined_data)

for (i, file_name) in enumerate(combined_data):
    if i == 3000: break
    target_dir = dev_dir if i < 1500 else test_dir
    shutil.move(os.path.join(source_dir, file_name), target_dir)
