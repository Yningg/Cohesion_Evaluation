"""
This script is used to write the bat file for the ST-Exa algorithm
Format:
@echo off
.\STExa.exe "D:\Cohesion_Evaluation\Input_Datasets\ST-Exa_Dataset\BTW17\tru-BTW17.txt" "D:\Cohesion_Evaluation\Original_Output\ST-Exa_Results\BTW17\4295_1_10.txt" 4295 1 10 1000

"""

import os

# Path to the dataset
dataset_name = ["BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"]
data_path = "D:\\Cohesion_Evaluation\\Input_Datasets\\ST-Exa_Dataset\\"



# Path to the exe file
exe_path = "D:\\Cohesion_Evaluation\\Representative_Algorithms\\ST-Exa\\"
exe_name = "STExa.exe"

# Parameters 
time_limit = 1000
lower_bound = [1, 11, 21, 31, 41, 51, 61, 71]
upper_bound = [10, 20, 30, 40, 50, 60, 70, 80]



# Write the bat file and store all the commands in one file
for dataset in dataset_name:
    
    # Path to query node file
    query_node_file = "D:\\Cohesion_Evaluation\\Original_Datasets\\Query_Nodes\\" + dataset + "_mapped_query_node.txt"

    # Path to the Result file
    result_fir = "D:\\Cohesion_Evaluation\\Algorithm_Output\\ST-Exa_Results\\" + dataset + "\\"

    # Create the result directory if it does not exist
    if not os.path.exists(result_fir):
        os.makedirs(result_fir)

    dataset_file = data_path + dataset + "\\" + "tru-" + dataset + ".txt"
    
    querynodes = []

    with open(query_node_file, 'r') as f:
        for line in f:
            querynodes.append(line.strip())


    print(f"Query nodes: {querynodes}")


    with open(exe_path + "STExa_" + dataset + ".bat", 'w') as bat_file:
        bat_file.write("@echo off\n")
        
        for query_node in querynodes:
            for lb, ub in zip(lower_bound, upper_bound): 
                result_file = result_fir + query_node + "_" + str(lb) + "_" + str(ub) + ".txt"
                
                bat_file.write(f'.\\{exe_name} "{dataset_file}" "{result_file}" {query_node} {lb} {ub} {time_limit}\n')
                print(f'command: .\\{exe_name} "{dataset_file}" "{result_file}" {query_node} {lb} {ub} {time_limit}')
    bat_file.close()