import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class ApproxPageRank{
	
	private static float alpha = (float)0.3;
    private static double epsilon = 0.00001;	
    private static String seed;
    private static String inputPath;
    private static Map<String, Float> pageRank;
    private static Map<String, Float> residual;
    private static void updateResidual(String key, float r_u, float weight) {
        if (residual.containsKey(key)) {
            residual.put(key, residual.get(key) + (1 - alpha) * r_u * weight);
        } else {
            residual.put(key, (1 - alpha) * r_u * weight);
        }
    }
    private static void push(String node, String[] nodeNeighbors, float r_u, int degree) {
        if (pageRank.containsKey(node)) {
            pageRank.put(node, pageRank.get(node) + alpha * r_u);
        } else {
            pageRank.put(node, alpha * r_u);
        }
        updateResidual(node, r_u, (float) 0.5);
        residual.put(node, (float) ((1 - alpha) * r_u * 0.5));
        for (int i = 0; i < nodeNeighbors.length; i++) {
        	updateResidual(nodeNeighbors[i], r_u, (float)(0.5 / degree));
        }
    }

		
	public static void main(String[] args) throws IOException{
		inputPath = args[0];
        seed = args[1];
        alpha = Float.parseFloat(args[2]);
        epsilon = Double.parseDouble(args[3]);

        pageRank = new HashMap<String, Float>();
        residual = new HashMap<String, Float>();

        residual.put(seed, (float)1.0);

        boolean flag = true;
        while (flag) {
        	flag = false;
            BufferedReader reader = new BufferedReader(new FileReader(inputPath));
            String line = null;

            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split("\t");
                String node = tokens[0];
                String[] nodeNeighbors;
                nodeNeighbors = Arrays.copyOfRange(tokens,1,tokens.length);
                float r_u = residual.containsKey(node)? residual.get(node): 0;
                int degree = nodeNeighbors.length;
                if(r_u/degree > epsilon){
                	push(node,nodeNeighbors,r_u,degree);
                    flag = true;
                }
            }
            
            reader.close();
        }
        for (Map.Entry<String, Float> entry : pageRank.entrySet()) {
            System.out.println(entry.getKey() + "\t" + entry.getValue());
        }
        
	}
	
}
	