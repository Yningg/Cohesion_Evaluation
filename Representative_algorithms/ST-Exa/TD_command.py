"""
This script is written to generate the bat file for executing the TD.exe file
format: .\TD.exe "D:\Cohesion_Evaluation\Input_Datasets\STExa_dataset\Crawled_Dataset26\Crawled_Dataset26.txt" "D:\Cohesion_Evaluation\Input_Datasets\STExa_dataset\Crawled_Dataset26\tru-Crawled_Dataset26.txt"
"""

import os

dataset_name = ["Crawled_Dataset26", "Crawled_Dataset144", "Chicago_COVID", "BTW17"]

# Path to the dataset
data_path = "D:\\Cohesion_Evaluation\\Input_Datasets\\ST-Exa_Dataset\\"

# Path to the exe file
exe_path = "D:\\Cohesion_Evaluation\\Representative_Algorithms\\ST-Exa\\"


for dataset in dataset_name:
    dataset_file = data_path + dataset + "\\" + dataset + ".txt"
    result_file = data_path + dataset + "\\tru-" + dataset + ".txt"

    exe_name = "TD.exe"

    with open(exe_path + "TD_" + dataset + ".bat", 'w') as bat_file:
        bat_file.write("@echo off\n")
        bat_file.write(f'.\\{exe_name} "{dataset_file}" "{result_file}"')
        print(f'command: .\\{exe_name} "{dataset_file}" "{result_file}"')
        bat_file.close()
