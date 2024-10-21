package com.heu.cswg.test;

import java.util.List;
import java.util.Arrays;
import Utils.StringFactory;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Map.Entry;
import java.util.stream.Gatherer.Integrator.Greedy;

import com.heu.cswg.model.Graph;
import com.heu.cswg.model.Vertex;
import com.heu.cswg.algorithm.*;
import com.heu.cswg.file.*;


public class Test_batch {
	
	/*
	 * 2018-05-21 LiuChiming
	 */
	
	public static  void firsttest()
	{	
		String dataset_name = "BTW17"; // "BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"
		String dataset_dir = "./Input_Datasets/I2ACSM_Dataset/";
		String query_node_dir = "./Input_Datasets/Query_nodes/";
		String results_path = "./Original_Output/CSD_Results/CSD_results_" + dataset_name + ".txt";

		String query_node_path = query_node_dir + dataset_name + "_query_node.txt";
		String graph_path = dataset_dir + dataset_name + "_non_attributed.txt";


		// Read the query node list
		List<Long> query_node_list = FileConverter.FileToQueryNodeList(query_node_path);

		Graph G = FileConverter.FileToGraph(graph_path);//Read dataset
		System.out.println("Initial vertex number:" + G.getVertexSize());
		Initialization ob = new Initialization();
		ob.edgeWeightInit(G); // Assign weights to edges
		System.out.println("Initialization finished");
		/*
			 // Test weights
		float union = ob.unionNum(G.getVertexMap().get(0), G.getVertexMap().get(2));
		float insert = ob.support(G.getVertexMap().get(0), G.getVertexMap().get(2));
		System.out.println("Set: "+union+","+insert);
		System.out.println("Weights: "+G.getVertexMap().get(0).getNeighborWeight().get(2));
		*/

		// Set the parameters
		List<Integer> d_list = Arrays.asList(1, 2, 3, 4, 5, 6); // Refer to paper: distance from the query node
		List<Integer> k_list = Arrays.asList(1, 2, 3, 4, 5, 6); // Refer to paper: k-truss
		List<Float> p_list = Arrays.asList(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f); // Refer to paper: influence threshold

		// Generate the parameter combinations for d, k and p
		List<ParameterCombination> combinations = new ArrayList<>();

        // Generate the combinations
        for (Integer d : d_list) {
            for (Integer k : k_list) {
                for (Float p : p_list) {
                    combinations.add(new ParameterCombination(d, k, p));
                }
            }
        }

		/* For each query node
		1. Iterate over the combinations
		2. Store the best final score and its corresponding community nodes and parameters
		3. Store the results for each query node into a txt file: Query node\tBest score\tBest parameters\tBest community  */
		List<String> results = new ArrayList<>();

		for (Long qId: query_node_list) {
			long startTime=System.currentTimeMillis();  // Record the start time

			for (ParameterCombination combination : combinations) {
				int d = combination.getD();
				int k = combination.getK();
				float p = combination.getP();
				
				Preproccess o = new Preproccess();
				Greedy o1 = new Greedy();

				
				List<Vertex> dlist = o.findDVertex(G, G.getVertexMap().get(qId), d); // Find the vertex set that is d away from the query node
				Graph G0 = o.inducedSubGraph(dlist, G); // build the subgraph obatined by dlist

				G0 = o.influenceMaintain(G0, (float) p, G.getVertexMap().get(qId));
			
				float score = o1.function(G0, G,G.getVertexMap().get(qId));
				System.out.println("Current G0 score: "+score);
				System.out.println("The number of nodes generated for the first time: "+ dlist.size()); 
				o.kdTrussMaintain(G0, G.getVertexMap().get(qId),k,d);//cut the obtained subgraph into (k, d)-truss
				System.out.println("The number of graph nodes after first trimming: " + G0.getVertexMap().size());
				
				List<Vertex> community = new ArrayList<>();
                List<Long> community_vertex_list = new ArrayList<>();
                List<Object> parameters = Arrays.asList(d, k, p);

				if (G0.getVertexMap().size() == 0) {
					System.out.println("The number of graph nodes after first trimming is 0");
					continue;
				}
				else
				{
					Graph A = o1.greedyWeight(G0, G, k, d, G0.getVertexMap().get(qId));//candidate graph search G' using greedy algorithm GCSM(H)
				
					// if the result graph has nodes, then calculate the score and compare with the best score
					if (A.getVertexIdList().size() > 0) {
						float function_score = o1.function(A, G, G.getVertexMap().get(qId));
						community = A.getVertexIdList();

						// Get vertex id list from the community
						for (Vertex vertex : community) {
							community_vertex_list.add(vertex.getId());
						}

						results.add(qId + "\t" + function_score + "\t" + parameters + "\t" + community_vertex_list);	
					}
					else
					{
						results.add(qId + "\t" + 0 + "\t" + parameters + "\t" + community_vertex_list);
					}
				}
			}
			// Store the the best score, best parameters, and best community for each query node
			long endTime = System.currentTimeMillis(); // Record the end time
			System.out.println("Running time: " + (endTime-startTime) + "ms");
			System.out.println("----------------------");
		}

		// Write the results into a txt file, with the first line: Query node\tBest score\tBest parameters\tBest community
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(results_path))) {
            // Write the first line
            // writer.write("Query node, Best score, Best parameters, Best community");
            writer.newLine();
            // Write the results
            for (String result : results) {
                writer.write(result);
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
	}
	
	public static void main(String[] args)
	{
		firsttest();
		System.out.println("-----End------");
		
	}

}