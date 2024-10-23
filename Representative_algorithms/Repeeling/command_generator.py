"""
This script is used to write the bat file for the Repeeling algorithm
Format:
@echo off
repeel-all.exe "D:\Cohesion_Evaluation\Input_Datasets\Repeeling_Dataset\Crawled_Dataset26\Crawled_Dataset26.txt" "D:\Cohesion_Evaluation\Original_Output\Repeeling_Results\4297_0.txt" 2600000 800 1 1 4297
repeel-all.exe "D:\Cohesion_Evaluation\Input_Datasets\Repeeling_Dataset\Crawled_Dataset26\Crawled_Dataset26.txt" "D:\Cohesion_Evaluation\Original_Output\Repeeling_Results\4297_1.txt" 2600000 800 2 1 4297
"""

import os


# Path to the exe file
exe_file = "D:\\Cohesion_Evaluation\\Representative_Algorithms\\Repeeling\\"
exe_name = "repeel-all.exe"

# Path for storing the bat file
bat_file_path = "D:/Cohesion_Evaluation/Representative_Algorithm/Repeeling/"

for dataset_name in ["BTW17", "Chicago_COVID", "Crawled_Dataset26", "Crawled_Dataset144"]:
    # Path to the dataset
    data_path = "D:\\Cohesion_Evaluation\\Repeeling_Dataset\\" + dataset_name + "\\" + dataset_name + ".txt"

    # Path to query node file
    query_node_file = "D:/Cohesion_Evaluation/Input_Datasets/Query_Nodes/" + dataset_name + "_mapped_query_node.txt"

    # Path to the Result file
    result_fir = "D:\\Cohesion_Evaluation\\Algorithm_Output\\Repeeling_Results\\" + dataset_name + "\\"


    querynodes = []
    with open(query_node_file, 'r') as f:
        for line in f:
            querynodes.append(line.strip())
    print(f"Query nodes: {querynodes}")

    # Create the result directory if it does not exist
    if not os.path.exists(result_fir):
        os.makedirs(result_fir)
    
    if dataset_name == "BTW17":
        # Parameters for BTW17
        window_size = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000]
        slide_percentage = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        k_c = [0, 1]
        k_f = [0, 1]
    elif dataset_name in ["Crawled_Dataset26", "Crawled_Dataset144"]:
        # Parameters for Crawled_Dataset26 and Crawled_Dataset144
        window_size = [100000, 500000, 1000000, 1500000, 2000000, 2500000]
        slide_percentage = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
        k_c = [0, 1, 2, 3]
        k_f = [0, 1, 2, 3]
    elif dataset_name == "Chicago_COVID":
        # Parameters for Chicago_COVID
        window_size = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]
        slide_percentage = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        k_c = [0, 1, 2, 3]
        k_f = [0, 1, 2, 3]
    else:
        print("Dataset not found")
        exit(1)



    # Write the bat file and store all the commands in one file
    with open(bat_file_path + dataset_name + ".bat", 'w') as bat_file:
        bat_file.write("@echo off\n")
        for query_node in querynodes:
            for w in window_size:
                for s in slide_percentage:
                    slide_size = int(w * s)
                    for c in k_c:
                        for f in k_f:
                            result_file = result_fir + query_node + "_" + str(w) + "_" + str(slide_size) + "_" + str(c) + "_" + str(f) + ".txt"
                            bat_file.write(f'{exe_name} "{data_path}" "{result_file}" {w} {slide_size} {c} {f} {query_node}\n')