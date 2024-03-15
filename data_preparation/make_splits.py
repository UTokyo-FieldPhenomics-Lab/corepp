import os
from pprint import pprint

val_split = ["R1-1",
             "R1-2",
             "R1-3",
             "R1-4",
             "R1-5",
             "R1-6",
             "R1-7",
             "R1-8",
             "R1-9",
             "R1-10",
             "R2-5",
             "3R3",
             "2R2",
             "4R3",
             "5R3",
             "3R5-2"
            ]


test_split = ["R12",
             "R13-9",
             "R13-10",
             "2R3",
             "3R2",
             "4R9",
             "5R2"
            ]

data_root = "./data/potato"
augm_root = "./data/potato_augmented"
num_aug = 10

all_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
val_folders = [folder for folder in all_folders if any(folder.startswith(prefix) for prefix in val_split)]
test_folders = [folder for folder in all_folders if any(folder.startswith(prefix) for prefix in test_split)]

set1 = set(all_folders)
set2 = set(val_folders)
set3 = set(test_folders)
train_folders_1 = list(set1 - set2)
train_folders = list(set(train_folders_1) - set3)

train_aug_folders = [f"{folder}_{i:02d}" for folder in train_folders for i in range(10)]
val_aug_folders = [f"{folder}_{i:02d}" for folder in val_folders for i in range(10)]

print("\r\nTrain folders:")
pprint(sorted(train_folders))
print("\r\nTrain Aug folders:")
pprint(sorted(train_aug_folders))
print("\r\nVal folders:")
pprint(sorted(val_folders))
print("\r\nTest folders:")
pprint(sorted(test_folders))