"""
Measuring the cohesiveness of a subgraph H in a directed multidigraph G using five measures
"""
import numpy as np
from collections import defaultdict


def time_decay(t_cur, t_i, rate, method):
    time_diff = t_cur - t_i

    if method == "exp":
        return np.exp(-rate * time_diff) 
    elif method == "poly":
        return 1 / ((time_diff + 1) ** rate)
        

# Calculate the excitation degree of node i up to time t in subgraph H
def excitation_degree(t, sentiment, activities, rate, method):
    degree = 1

    sentimental_activities = [activity for activity in activities if activity[3] != 0]
    if not sentimental_activities:
        return degree
        
    timestamp_list = np.array([activity[2] for activity in sentimental_activities])
    sentiment_list = np.array([activity[3] for activity in sentimental_activities])

    sign_values = np.sign(sentiment_list * sentiment) # Identify the polarity of the sentiment
    decay_values = np.array([time_decay(t, timestamp, rate, method) for timestamp in timestamp_list])

    degree += np.dot(sign_values, decay_values)
    
    return max(0, degree)


# Calculate the elicited sentiment Esenti
def ESenti(t, sentiment, activity_list, rate, method):
    if sentiment == 0:
        return 0
    return sentiment * excitation_degree(t, sentiment, activity_list, rate, method)


# Calculate the three ATG-S measures of subgraph H in graph G at time t_cur
def ATGS(tadj_list, tadj_sublist, u, t_cur, rate, method, mutual_enjoyment):
    EI_value, SIT_value, CED_value = 0, 0, 0
    enjoyment_rest = 0
    tadj_list_u = tadj_list[u]
    tadj_sublist_u = tadj_sublist[u]

    # Calculate the Enjoyment Index (EI) of node u in subgraph H at time t_cur
    ESenti_values_H = np.array([ESenti(timestamp, sentiment, tadj_sublist_u[:i], rate, method) for i, (_, _, timestamp, sentiment) in enumerate(tadj_sublist_u)])
    time_decay_values_H = np.array([time_decay(t_cur, timestamp, rate, method) for (_, _, timestamp, _) in tadj_sublist_u])
    EI_value = np.dot(ESenti_values_H, time_decay_values_H)


    # Calculate the Sentimental interaction tendency (SIT) of node i in subgraph H at time t_cur
    H_ij_activities = defaultdict(list)
    
    # Find mutually interacting user pairs in subgraph H
    entire_H_user_pair = set((edge[0], edge[1]) for edge in tadj_sublist_u if edge[0] != edge[1])
    H_user_pair = set()

    for user_pair in entire_H_user_pair:
        if (user_pair[1], user_pair[0]) in entire_H_user_pair:
            H_user_pair.add((min(user_pair[0], user_pair[1]), max(user_pair[0], user_pair[1])))
    
    if H_user_pair:
        for (u_i, u_j, timestamp, sentiment) in tadj_sublist_u:
            pair = (min(u_i, u_j), max(u_i, u_j))
            if pair in H_user_pair:
                H_ij_activities[pair].append((u_i, u_j, timestamp, sentiment))
    
        for (u_i, u_j) in H_user_pair:
            if (u_i, u_j) in mutual_enjoyment.keys():
                SIT_value += mutual_enjoyment.pop((u_i, u_j)) # Avoid repeated calculation
            else:
                activities_H_ij_cur = H_ij_activities[(u_i, u_j)]
                ESenti_values_ij = np.array([ESenti(timestamp, sentiment, activities_H_ij_cur[:i], rate, method) for i, (_, _, timestamp, sentiment) in enumerate(activities_H_ij_cur)])
                time_decay_values_ij = np.array([time_decay(t_cur, timestamp, rate, method) for (_, _, timestamp, _) in activities_H_ij_cur])
                SIT_temp = np.dot(ESenti_values_ij, time_decay_values_ij)
                SIT_value += SIT_temp
                mutual_enjoyment[(u_i, u_j)] = SIT_temp


    # Calculate the Comparative Enjoyment Degree(CED) of node i in subgraph H at time t_cur
    # Find tadjs of node u in graph G that are not in subgraph H
    filtered_tadj_list = [activity for activity in tadj_list_u if activity not in tadj_sublist_u]
    ESenti_values_G_H = np.array([ESenti(timestamp, sentiment, tadj_list_u[:i], rate, method) for i, (_, _, timestamp, sentiment) in enumerate(filtered_tadj_list)])
    time_decay_values_G_H = np.array([time_decay(t_cur, timestamp, rate, method) for (_, _, timestamp, _) in filtered_tadj_list])
    enjoyment_rest = np.dot(ESenti_values_G_H, time_decay_values_G_H)
    
    CED_value = EI_value - enjoyment_rest

    return EI_value, SIT_value, CED_value, mutual_enjoyment


# Calculate the GI-S measures of subgraph H in graph G at time t_cur
def GIS(tadj_sublist, nodes_num):
    interaction_activities, total_activities = set(), set()
    GID_value, GIP_value = 0, 0

    if nodes_num > 1:
        for u_i, edges in tadj_sublist.items():
            for (u_i, u_j, timestamp, _) in edges:
                total_activities.add((u_i, u_j, timestamp))
                if u_i != u_j:
                    interaction_activities.add((u_i, u_j, timestamp))

        interaction_activities_num = len(interaction_activities)
        total_activities_num = len(total_activities)

        if total_activities_num > 0:
            GIP_value = interaction_activities_num / total_activities_num
        
        GID_value = interaction_activities_num / (nodes_num * (nodes_num - 1))

    return GIP_value, GID_value


def cohesiveness_dim(tadj_list, tadj_sublist, t_cur, rate, method):
    EI_list, SIT_list, CED_list = [], [], []
    mutual_enjoyment = {} # Store the mutual enjoyment between node i and node j, avoid repeated calculation
    N = len(tadj_sublist)
    
    for u_i in tadj_sublist.keys():
        EI_value, SIT_value, CED_value, mutual_enjoyment = ATGS(tadj_list, tadj_sublist, u_i, t_cur, rate, method, mutual_enjoyment)

        EI_list.append(EI_value)
        SIT_list.append(SIT_value)
        CED_list.append(CED_value)
        
    EI_avg = np.sum(EI_list) / N
    SIT_avg = np.sum(SIT_list) / N
    CED_avg = np.sum(CED_list) / N
    GIP, GID = GIS(tadj_sublist, N)

    return [float(EI_avg), float(SIT_avg), float(CED_avg), GIP, GID]