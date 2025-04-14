package csd;

import java.util.*;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;


import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import javax.management.Query;

import csd.Algos.DirectedAlgo.dcore.DcoreDecomposition;
import csd.Algos.DirectedAlgo.index.*;
import csd.util.DataReader;

import csd.util.Abstracts.Tables;

/**
 * @author Zhongran
 * @date Jun 29, 2017
 * to show how to build an index
 */
public class Test {

    public static void main(String [] args){
        String dataName = "BTW17"; // "BTW17", "Chicago_COVID", "Crawled_Dataset144", "Crawled_Dataset26"
        String query_node_dir = "D:/Cohesion_Evaluation/Original_Datasets/Query_nodes/";
        String query_node_file = query_node_dir + dataName + "_mapped_query_node.txt";
        String results_path = "D:/Cohesion_Evaluation/Algorithm_Output/CSD_Results/" + "CSD_results_" + dataName + ".txt";

        // Read query node list directly from file
        List<Integer> query_nodes = new ArrayList<>();
        try {
            Scanner scanner = new Scanner(new java.io.File(query_node_file));
            while (scanner.hasNextInt()) {
                query_nodes.add(scanner.nextInt());
            }
            scanner.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

       
        // DataReader reader = new DataReader(dataName, 100);
        DataReader reader = new DataReader(dataName); //Read data without sampling
        reader.read(); //Read data

        DcoreDecomposition decomp = new DcoreDecomposition(reader.getIn(), reader.getOut()); // Do decomposition

        //Tables t = new NestIdx(decomp);
        Tables t = new PathIdx(decomp);
        //Tables t = new UnionIdx(decomp);
        t.buildIndex(); // build a nested idx
        
        // Set the parameters for query
        List<Integer> k_list = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        List<Integer> l_list = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        List<Entry<Integer, Integer>> combinations = new ArrayList<>();

        // Generate the combinations
        for (Integer k : k_list) {
            for (Integer l : l_list) {
                combinations.add(new SimpleEntry<>(k, l));
            }
        }

        /* For each query node
		1. Iterate over the combinations
		2. Store the the non-empty results for each combination
		3. Store the results for each query node into a txt file: Query node\tParameters\tCommunity  */
        List<String> results = new ArrayList<>();

        for (Integer qNode: query_nodes) {
            for (Entry<Integer, Integer> comb : combinations) {
                int k = comb.getKey();
                int l = comb.getValue();
                int [] queryResult = t.query(qNode, k, l);
                if(queryResult != null) {
                    List<Integer> parameters = Arrays.asList(k, l);
                    // Connect queryResult to a list
                    List<Integer> community_nodes = new ArrayList<>();
                    for (int i = 0; i < queryResult.length; i++) {
                        community_nodes.add(queryResult[i]);
                    }
                    results.add(qNode + "\t" + parameters + "\t" + community_nodes);
                }
                else{
                    System.out.println("No community found for query node " + qNode + " with parameters " + k + " and " + l);
                    results.add(qNode + "\t" + Arrays.asList(k, l) + "\t" + new ArrayList<Integer>());
                }
            }
        }

        // Write the results to a file
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(results_path))) {
            for (String result : results) {
                writer.write(result);
                writer.newLine();
            }
        } 
        catch (IOException e) 
        {
            e.printStackTrace();
        }

    }
}
