import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;



public class LR{
	public static void main(String[] args) throws IOException{
		if (args.length < 6) {
            System.exit(-1);
        }
		
		init(args);
		train();
		predict();		
	}
	
	final static double overflow = 20;
	final static String[] Label = {"nl", "el", "ru", "sl", "pl", "ca", "fr", "tr", "hu", "de", "hr", "es", "ga", "pt"};
	final static int labelNum = 14;
	static private HashMap<String, Integer> labelMap = new HashMap<String, Integer>();
	private static int vocaSize;
	private static double learningRate;
	private static double regCo;
	private static int maxIter;
	private static int trainSize;
	private static String testFile;
	//A will record the value of k last time B[j] was updated
	private static int[][] A;
	private static double[][] B;
	private static int k = 0;
	private static HashMap<Integer, Integer> tokenMap;
	

	//private static double[] LCL;
	//initiation
	
	private static void init(String[] args){
		vocaSize = Integer.parseInt(args[0]);
		learningRate = Double.parseDouble(args[1]);
		regCo = Double.parseDouble(args[2]);
		maxIter = Integer.parseInt(args[3]);
		trainSize = Integer.parseInt(args[4]);
		testFile = args[5];
		A = new int[labelNum][vocaSize];
		B = new double[labelNum][vocaSize];
		tokenMap = new HashMap<Integer, Integer>();
		

		//LCL = new double[labelNum];
		/*
		for (int i=0;i<labelNum;i++){
			LCL[i] = 0;
		}
		*/
		for(int i=0;i<labelNum;i++){
			labelMap.put(Label[i], i);
		}
	}
	
	//tokenize
	private static Vector<String> tokenizeDoc(String cur_doc) {
        String[] words = cur_doc.split("\\s+");
        Vector<String> tokens = new Vector<String>();
        for (int i = 0; i < words.length; i++) {
        	words[i] = words[i].replaceAll("\\W", "");
        	if (words[i].length() > 0) {
        		tokens.add(words[i]);
        	}
        }
        return tokens;
	}
	private static int myhashFunc(String str){
		int id = str.hashCode() % vocaSize;
		if(id < 0){
			id += vocaSize;
		}
		return id;
	}
	
	//
	private static double probTrans(double score) {
        return 1.0 - 1.0 / (1.0 + Math.exp(score));
    }
	private static double sigmoid(double score) {
		if (score > overflow) score = overflow;
		else if (score < -overflow) score = -overflow;
		double exp = Math.exp(score);
		return exp / (1 + exp);
		}
	
	private static void probCalc(Vector<String> tokens, double[] pr){
		for (String token: tokens){
			int id = myhashFunc(token);
			if(tokenMap.containsKey(id)){
				tokenMap.put(id, tokenMap.get(id) + 1);
			}
			else{
				tokenMap.put(id,  1);
			}
			for (int i=0;i<labelNum;i++){
				pr[i] += B[i][id];
			}
		}
		for (int i=0;i<labelNum;i++){
			pr[i] = probTrans(pr[i]);
			//pr[i] = sigmoid(pr[i]);
		}
	
	}
	
	private static void updateAB(double[] pr, boolean[] labelExist, double curlearningRate){
		for (Map.Entry<Integer, Integer> entry:tokenMap.entrySet()){
			int j = entry.getKey();
			int count = entry.getValue();
			for (int i=0; i<labelNum;i++){
				B[i][j] *=  Math.pow(1 - 2 * curlearningRate * regCo, k - A[i][j]);
				if(labelExist[i]){
					B[i][j] += curlearningRate * (1 - pr[i]) * count;
					//LCL[i] += (1-pr[i])*count;
					
				}
				else{
					B[i][j] += curlearningRate * (0 - pr[i]) * count;
					//LCL[i] += (0 - pr[i])*count;
					
				}
				A[i][j] = k;
				
				
			}
		}
	}
	private static void modifyB(double curlearningRate){
		for (int i=0;i<labelNum;i++){
			for(int j=0;j<vocaSize;j++){
				B[i][j] *=  Math.pow(1 - 2 * curlearningRate * regCo, k - A[i][j]);
			}
		}
	}
	
	//train LR with Efficient regularized SGD
	private static void train() throws IOException{
		BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
		String line = bufferedReader.readLine();
		//double totalLCL = 0;
		double curlearningRate = learningRate;
		while(line  != null){
			for(int t=0;t<maxIter;t++){
				//update learning rate
				curlearningRate = learningRate / Math.pow(t + 1.0, 2);
				for(int lineNum = 0;lineNum<trainSize && line!=null;lineNum++){
					k += 1;
					String[] parseLine = line.split("\t");
					String[] labels = parseLine[0].split(",");
					boolean[] labelExist = new boolean[labelNum];
					double[] pr = new double[labelNum];
					for(String label : labels){
						labelExist[labelMap.get(label)] = true;
					}
					Vector<String> tokens = tokenizeDoc(parseLine[1]);
					tokenMap.clear();
					probCalc(tokens, pr);
					updateAB(pr, labelExist, curlearningRate);
					line = bufferedReader.readLine();
				}
				/*
				for(int i=0;i<labelNum;i++){
					totalLCL += LCL[i];
				}
				System.out.println(String.valueOf(totalLCL));
				*/
			}
		}
		modifyB(curlearningRate);
		bufferedReader.close();
	}
	
	//predict 
	private static void predict() throws IOException{
		BufferedReader bufferedReader = new BufferedReader(new FileReader(testFile));
		String line = null;
		//int accCount = 0, totalCount = 0;
		while ((line = bufferedReader.readLine()) != null) {
			String[] parseLine = line.split("\t");
			Vector<String> tokens = tokenizeDoc(parseLine[1]);
			//String[] labels = parseLine[0].split(",");
            //ArrayList<Integer> curlabels = new ArrayList<Integer> ();
            /*
			for (String label : labels) {
                if (labelMap.containsKey(label)) {
                    int i = labelMap.get(label);
                    curlabels.add(i);
                }
            }*/
			double[] pr = new double[labelNum];
			String res = "";
			for(String token : tokens){
				for (int i = 0; i < labelNum; i++) {
                    pr[i] += B[i][myhashFunc(token)];
                }
			}
			int i = 0;
			for (; i < labelNum - 1; i++) {
				pr[i] = probTrans(pr[i]);
	            res = String.format("%s%s\t%f,", res, Label[i], pr[i]);
            }
			pr[i] = probTrans(pr[i]);
            res = String.format("%s%s\t%f", res, Label[i], pr[i]);
			System.out.println(res);
			/*
            totalCount++;
            for (int label : curlabels) {
                if (pr[label] >= 0.5) {
                	accCount++;
                	break;
                }
            }*/
		}
		//System.out.println("Accuracy:" + (double)accCount / totalCount);
		bufferedReader.close();
		
	}
	
}