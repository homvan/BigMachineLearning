import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class DataLoader {

  private String dataFile;

  public DataLoader(String dataFile) {
    this.dataFile = dataFile;
  }

  public int[][] load(int part, int numParts) {
    ArrayList<String> lines = new ArrayList<String>();
    try {
      FileReader fr = new FileReader(this.dataFile);
      BufferedReader br = new BufferedReader(fr);
      int i = 0;
      while (true) {
        String line = br.readLine();
        if (line == null) {
          break;
        } else if (i % numParts == part) {
          lines.add(line);
        }
        i ++;
      }
      br.close();
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(1);
    }
    int[][] w = new int[lines.size()][];
    for (int i = 0; i < lines.size(); i ++) {
      String[] tokens = lines.get(i).split(",");
      w[i] = new int[tokens.length];
      for (int j = 0; j < tokens.length; j ++) {
        w[i][j] = Integer.parseInt(tokens[j]);
      }
    }
    return w;
  }

}
