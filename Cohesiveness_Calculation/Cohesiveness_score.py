"""
This script is used to implement the calculation of cohesiveness score of a subgraph H in a directed multidigraph G
"""
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import tqdm
import time
from collections import defaultdict

seed = 2024 # Seed
random.seed(seed)
np.random.seed(seed)


# Time decay function:
def time_decay(t_cur, t_i, rate, method):
    # print(f"Time diff: {t_cur - t_i}")
    # print(f"Time decay: {np.exp(-beta * (t_cur - t_i))}")
    time_diff = t_cur - t_i

    if method == "exp":
        if type(time_diff) == int:
            return np.exp(-rate * time_diff)
        else:
            return np.exp(-rate* (t_cur - t_i).seconds)
    elif method == "poly":
        if type(time_diff) == int:
            return 1 / pow(time_diff + 1, rate)
        else:
            return 1 / pow((t_cur - t_i).seconds + 1, rate)


# Identify the polarity of the sentiment
def sgn(previous_sentiment, sentiment):
    if previous_sentiment * sentiment > 0:
        return 1
    elif previous_sentiment * sentiment < 0:
        return -1
    else:
        return 0


# Calculate the excitation degree of node i up to time t in subgraph H
def excitation_degree(t, sentiment, activities, rate, method):
    degree = 1

    # Unvectorized version
    # for u_i, u_j, data in activities:
    #     degree += sgn(data['sentiment'], sentiment) * time_decay(t, data['timestamp'], rate, method) # type: ignore

    # Vectorized version
    activities_num = len(activities)
    sign_values, decay_values = [0] * activities_num, [0] * activities_num
    for i, (_, _, timestamp1, sentiment1) in enumerate(activities):
        sign_values[i] = sgn(sentiment1, sentiment)
        decay_values[i] = time_decay(t, timestamp1, rate, method) # type: ignore

    degree += np.dot(np.array(sign_values), np.array(decay_values))
    
    return degree

# Calculate the elicited seniment Esenti(a_ij^t)
def ESenti(t, sentiment, activity_list, rate, method):

    elicited_sentiment = sentiment * excitation_degree(t, sentiment, activity_list, rate, method)
    
    return elicited_sentiment

"""
Calculate the ATGS score
1. Extract the user_pair from the tadj_list
2. Extract the H_ij activities of node i in subgraph H up to time t_cur
3. Calculate three scores of EL, SIT, and CED
"""
def ATGS(edge_stream, tadj_list, edge_substream, tadj_sublist, u, t_cur, rate, method, mutual_enjoyment):
    EL_value, SIT_value, CED_value = 0, 0, 0
    enjoyment_total = 0

    # Extract the H_ij activities of node u in subgraph H up to time t_cur
    # Initialize the defaultdict for activities
    H_ij_activities = defaultdict(list)

    # Create a set of user pairs
    H_user_pair = set((min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in tadj_sublist[u])

    # Iterate over the edges and accumulate activities
    for (u_i, u_j, timestamp, sentiment) in tadj_sublist[u]:
        pair = (min(u_i, u_j), max(u_i, u_j))
        if pair in H_user_pair:
            H_ij_activities[pair].append((u_i, u_j, timestamp, sentiment))
  

    # (Vectorized version) Calculate the Enjoyment Level (EL) of node u in subgraph H at time t_cur
    ESenti_values_H = np.array([ESenti(timestamp, sentiment, tadj_sublist[u][:i], rate, method) for i, (_, _, timestamp, sentiment) in enumerate(tadj_sublist[u])])
    time_decay_values_H = np.array([time_decay(t_cur, timestamp, rate, method) for (_, _, timestamp, _) in tadj_sublist[u]])
    EL_value = np.dot(ESenti_values_H, time_decay_values_H)

    # (Vectorized version) Calculate the Sentimental interaction tendency (SIT) of node i in subgraph H at time t_cur
    for (u_i, u_j) in H_user_pair:
        if (u_i, u_j) in mutual_enjoyment.keys():
            SIT_value += mutual_enjoyment.pop((u_i, u_j))
        else:
            activities_H_ij_cur = H_ij_activities[(u_i, u_j)]
            ESenti_values_ij = np.array([ESenti(timestamp, sentiment, activities_H_ij_cur[:i], rate, method) for i, (_, _, timestamp, sentiment) in enumerate(activities_H_ij_cur)])
            time_decay_values_ij = np.array([time_decay(t_cur, timestamp, rate, method) for (_, _, timestamp, _) in activities_H_ij_cur])
            SIT_temp = np.dot(ESenti_values_ij, time_decay_values_ij)
            SIT_value += SIT_temp
            mutual_enjoyment[(u_i, u_j)] = SIT_temp

    # (Vectorized version) Calculate the Comparative Enjoyment Degree(CED) of node i in subgraph H at time t_cur
    ESenti_values_G = np.array([ESenti(timestamp, sentiment, tadj_list[u][:i], rate, method) for i, (_, _, timestamp, sentiment) in enumerate(tadj_list[u])])
    time_decay_values_G = np.array([time_decay(t_cur, timestamp, rate, method) for _, _, timestamp, _ in tadj_list[u]])
    enjoyment_total = np.dot(ESenti_values_G, time_decay_values_G)
    
    CED_value = EL_value - (enjoyment_total - EL_value)

    return EL_value, SIT_value, CED_value, mutual_enjoyment


# Calculate the GIS of subgraph H in graph G at time t_cur
def GIS(tadj_sublist):
    # Extract the set of self-loop edges and interaction edges in subgraph H up to time t_cur
    interaction_activities, total_activities = set(), set()
    nodes_num = len(tadj_sublist.keys())
    GID_value, GIP_value = 0, 0

    if nodes_num > 1:
        for u_i in tadj_sublist.keys():
            for (u_i, u_j, timestamp, _) in tadj_sublist[u_i]:
                total_activities.add((u_i, u_j, timestamp))
                if u_i != u_j:
                    interaction_activities.add((u_i, u_j, timestamp))

        interaction_activities_num = len(interaction_activities)
        total_activities_num = len(total_activities)

        if total_activities_num > 0:
            GIP_value = interaction_activities_num / total_activities_num
        
        GID_value = interaction_activities_num / (nodes_num * (nodes_num - 1))

    return GIP_value, GID_value


def cohesiveness_dim(edge_stream, tadj_list, edge_substream, tadj_sublist, t_cur, rate, method):
    # Cut the edge stream and tadj_list to the current time
    edge_stream = {timestamp: edge_stream[timestamp] for timestamp in edge_stream if timestamp <= t_cur}
    tadj_list = {u: [edge for edge in tadj_list[u] if edge[2] <= t_cur] for u in tadj_list.keys()}

    
    N = len(tadj_sublist)
    EL_list, SIT_list, CED_list = [], [], []

    mutual_enjoyment = {} # A dict to store the mutual enjoyment between node i and node j, avoid repeated calculation
       
    for u_i in tadj_sublist.keys():
        EL_value, SIT_value, CED_value, mutual_enjoyment = ATGS(edge_stream, tadj_list, edge_substream, tadj_sublist, u_i, t_cur, rate, method, mutual_enjoyment)

        EL_list.append(EL_value)
        SIT_list.append(SIT_value)
        CED_list.append(CED_value)
        
    EL_avg = sum(EL_list) / N
    SIT_avg = sum(SIT_list) / N
    CED_avg = sum(CED_list) / N
    GIP, GID = GIS(tadj_sublist)

    return [EL_avg, SIT_avg, CED_avg, GIP, GID]
