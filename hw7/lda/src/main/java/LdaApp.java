import org.petuum.jbosen.PsApplication;
import org.petuum.jbosen.PsTableGroup;
import org.petuum.jbosen.table.IntTable;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoublePredicate;

import org.apache.commons.math3.special.*;

public class LdaApp extends PsApplication {

  private static final int TOPIC_TABLE = 0;
  private static final int WORD_TOPIC_TABLE = 1;
  
  private String outputDir;
  private int numWords;
  private int numTopics;
  private double alpha;
  private double beta;
  private int numIterations;
  private int numClocksPerIteration;
  private int staleness;
  private DataLoader dataLoader;
  private Random random;

  public LdaApp(String dataFile, String outputDir, int numWords, int numTopics,
                double alpha, double beta, int numIterations,
                int numClocksPerIteration, int staleness) {
    this.outputDir = outputDir;
    this.numWords = numWords;
    this.numTopics = numTopics;
    this.alpha = alpha;
    this.beta = beta;
    this.numIterations = numIterations;
    this.numClocksPerIteration = numClocksPerIteration;
    this.staleness = staleness;
    this.dataLoader = new DataLoader(dataFile);
    this.random = new Random();
  }

  public double logDirichlet(double[] alpha) {
		double sumLogGamma=0.0;
		double logSumGamma=0.0;
		for (double value : alpha){
			sumLogGamma += Gamma.logGamma(value);
			logSumGamma += value;
		}
		return sumLogGamma - Gamma.logGamma(logSumGamma);
	}

	public double logDirichlet(double alpha, int k) {
		return k * Gamma.logGamma(alpha) - Gamma.logGamma(k*alpha);
	}
	
	public double[] getRows(IntTable matrix, int columnId) {
		double[] rows = new double[this.numWords];
		for (int i = 0; i < this.numWords; i ++){
			rows[i] = (double) matrix.get(i, columnId);
		}
		return rows;
	}
	
	public double[] getColumns(int[][] matrix, int rowId){
		double[] cols = new double[this.numTopics];
		for (int i = 0; i < this.numTopics; i ++){
			cols[i] = (double) matrix[rowId][i];
		}
		return cols;
	}

	public double getLogLikelihood(IntTable wordTopicTable,
                                 int[][] docTopicTable) {
	  double lik = 0.0;
	  for (int k = 0; k < this.numTopics; k ++) {
		  double[] temp = this.getRows(wordTopicTable, k);
		  for (int w = 0; w < this.numWords; w ++) {
				 temp[w] += this.alpha;
		  }
		  lik += this.logDirichlet(temp);
		  lik -= this.logDirichlet(this.beta, this.numWords);
	  }
	  for (int d = 0; d < docTopicTable.length; d ++) {
		  double[] temp = this.getColumns(docTopicTable, d);
		  for (int k = 0; k < this.numTopics; k ++) {
			 temp[k] += this.alpha;
		  }
		  lik += this.logDirichlet(temp);
		  lik -= this.logDirichlet(this.alpha, this.numTopics);
	  }
	  return lik;
  }
  
  @Override
  public void initialize() {
    // Create global topic count table. This table only has one row, which
    // contains counts for all topics.
    PsTableGroup.createDenseIntTable(TOPIC_TABLE, staleness, numTopics);
    // Create global word-topic table. This table contains numWords rows, each
    // of which has numTopics columns.
    PsTableGroup.createDenseIntTable(WORD_TOPIC_TABLE, staleness, numTopics);
  }

  @Override
  public void runWorkerThread(int threadId) {
    int clientId = PsTableGroup.getClientId();

    // Load data for this thread.
    System.out.println("Client " + clientId + " thread " + threadId +
                       " loading data...");
    int part = PsTableGroup.getNumLocalWorkerThreads() * clientId + threadId;
    int numParts = PsTableGroup.getNumTotalWorkerThreads();
    int[][] w = this.dataLoader.load(part, numParts);

    // Get global tables.
    IntTable topicTable = PsTableGroup.getIntTable(TOPIC_TABLE);
    IntTable wordTopicTable = PsTableGroup.getIntTable(WORD_TOPIC_TABLE);
    
    // Initialize LDA variables.
    System.out.println("Client " + clientId + " thread " + threadId +
                       " initializing variables...");
    //Number of doc topic
    int[][] docTopicTable = new int[w.length][this.numTopics];
    //Topic label table
    int[][] z = new int[w.length][];
    for (int d = 0; d < w.length; d++){
    	int N = w[d].length;
    	z[d] = new int[N];
    	for(int i = 0; i < N; i ++){
    		int word = w[d][i];
    		int initTopic = random.nextInt(numTopics);
    		z[d][i] = initTopic;
    		docTopicTable[d][initTopic] += 1;
    		topicTable.inc(0, initTopic, 1);
    		wordTopicTable.inc(word, initTopic, 1);
    	}
    }
    
    
    
    // Global barrier to synchronize word-topic table.
    PsTableGroup.globalBarrier();

    // Do LDA Gibbs sampling.
    System.out.println("Client " + clientId + " thread " + threadId +
                       " starting gibbs sampling...");
    double[] llh = new double[this.numIterations];
    double[] sec = new double[this.numIterations];
    double totalSec = 0.0;
    for (int iter = 0; iter < this.numIterations; iter ++) {
      long startTime = System.currentTimeMillis();
      // Each iteration consists of a number of batches, and we clock
      // between each to communicate parameters according to SSP.
      for (int batch = 0; batch < this.numClocksPerIteration; batch ++) {
        int begin = w.length * batch / this.numClocksPerIteration;
        int end = w.length * (batch + 1) / this.numClocksPerIteration;
        // Loop through each document in the current batch.
        for (int d = begin; d < end; d ++) {
          int N = w[d].length;
          for(int i = 0; i < N; i++){
        	  int word = w[d][i];
        	  int topic = z[d][i];
        	  docTopicTable[d][topic] -= 1;
        	  wordTopicTable.inc(word, topic, -1);
        	  topicTable.inc(0, topic, -1);
        	  double[] p = new double[numTopics];
        	  for(int k = 0; k < numTopics; k++){
        		  p[k] = (wordTopicTable.get(word, k) + beta) / (topicTable.get(0, k) + numWords * beta) * (docTopicTable[d][k] + alpha);
        	  }
        	  for (int k = 1;k < numTopics; k++){
        		  p[k] += p[k - 1];
        	  }
        	  double u = random.nextDouble() * p[numTopics - 1];
        	  int newTopic;
        	  for(newTopic = 0; newTopic <numTopics; newTopic++){
        		  if(u < p[newTopic]){
        			  break;
        		  }
        	  }
        	  docTopicTable[d][newTopic] += 1;
        	  topicTable.inc(0, newTopic, 1);
        	  wordTopicTable.inc(word, newTopic, 1);
        	  z[d][i] = newTopic;
          }
        	
        }
        // Call clock() to indicate an SSP boundary.
        PsTableGroup.clock();
      }
      // Calculate likelihood and elapsed time.
      totalSec += (double) (System.currentTimeMillis() - startTime) / 1000; 
      sec[iter] = totalSec;
      llh[iter] = this.getLogLikelihood(wordTopicTable, docTopicTable);
      System.out.println("Client " + clientId + " thread " + threadId +
                         " completed iteration " + (iter + 1) +
                         "\n    Elapsed seconds: " + sec[iter] +
                         "\n    Log-likelihood: " + llh[iter]);
    }

    PsTableGroup.globalBarrier();

    // Output likelihood.
    System.out.println("Client " + clientId + " thread " + threadId +
                       " writing likelihood to file...");
    try {
      PrintWriter out = new PrintWriter(this.outputDir + "/likelihood_" +
                                        clientId + "-" + threadId + ".csv");
      for (int i = 0; i < this.numIterations; i ++) {
        out.println((i + 1) + "," + sec[i] + "," + llh[i]);
      }
      out.close();
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(1);
    }
    
    PsTableGroup.globalBarrier();
    
    // Output tables.
    if (clientId == 0 && threadId == 0) {
      System.out.println("Client " + clientId + " thread " + threadId +
                         " writing word-topic table to file...");
      try {
        PrintWriter out = new PrintWriter(this.outputDir + "/word-topic.csv");
        for (int i = 0; i < numWords; i ++) {
          out.print(wordTopicTable.get(i, 0));
          for (int k = 1; k < numTopics; k ++) {
            out.print("," + wordTopicTable.get(i, k));
          }
          out.println();
        }
        out.close();
      } catch (IOException e) {
        e.printStackTrace();
        System.exit(1);
      }
    }

    

    System.out.println("Client " + clientId + " thread " + threadId +
                       " exited.");
    System.out.println("Calculating tfidf...");
    //compute tfidf
    int[] df = new int[this.numWords];
    //total word num in each topic
    int[] wordNumTopic = new int[this.numTopics];
    for(int  i = 0; i < this.numWords; i++){
    	int tmp = 0;
    	for (int j = 0; j < numTopics; j++){
    		wordNumTopic[j] += wordTopicTable.get(i, j);
    		if(wordTopicTable.get(i, j) > 0){
    			tmp += 1;
    		}
    	}
    	df[i] = tmp;
    }
    
    for(int topic = 0; topic < this.numTopics; topic++){
    	Map<Integer, Double> wordTfidf = new HashMap<Integer, Double>();
    	for(int i = 0; i < this.numWords; i++){
    		if(wordTopicTable.get(i, topic) > 0){
    			double a = wordTopicTable.get(i, topic);
    			double b = wordNumTopic[topic];
    			double tf = a/b;
        		double tfidf = Math.log(tf) * Math.log(20.0/df[i]);
        		wordTfidf.put(i, tfidf);
    		}
    		
    		
    	}
    	System.out.println("Sorting...");
    	List<Entry<Integer, Double>> sortTfidf = new ArrayList<Entry<Integer, Double>>(wordTfidf.entrySet());
    	
    	Collections.sort(sortTfidf, new Comparator<Map.Entry<Integer, Double>>() {
			@Override
			public int compare(Entry<Integer, Double> o1, Entry<Integer, Double> o2) {
				// TODO Auto-generated method stub
				return o1.getValue().compareTo(o2.getValue());
			}
		});
    	int i = 1;
    	for(Map.Entry<Integer, Double> entry : sortTfidf){
    		if(i <= 5){
    			System.out.println("Topic: " + topic + "; Rank: " + i + "; Word: " + entry.getKey() + "; TFIDF: "+ entry.getValue().toString());
    			i += 1;
    		}
    	}
    }
    PsTableGroup.globalBarrier();
    
  }

  

}
