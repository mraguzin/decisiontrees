package com.mraguzin.decisiontrees;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import java.io.File;

/**
 *
 * @author mraguzin
 */
public class DecisionTrees {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Reader in;
        CSVParser records;
        Map<String, Integer> attributes;

        try {
            if (args.length < 2) {
                System.out.println("Upotreba: program podaci.csv novi_primjer");
                return;
            }
            System.out.println(args[0]);
            in = new FileReader(args[0]);
            records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(in);
            attributes = records.getHeaderMap();
        } catch (FileNotFoundException ex) {
            System.out.println(args[0] + " ne postoji!");
            return;
        } catch (IOException ex) {
            System.out.println("Greška pri čitanju csv-a");
            return;
        }

        String classAttribute = ""; // we assume the classification/result attribute
        // resides in the last csv column
        var it = attributes.entrySet().iterator();
        int i;
        for (i = 0; i < attributes.size(); ++i) {
            var pair = it.next();
            if (i == attributes.size() - 1) {
                classAttribute = pair.getKey();
            }
        }

        System.out.println(classAttribute);
        var tree = new DepthFirstTree(attributes, classAttribute,
                (x) -> !x.toLowerCase().contains("no"), records.getRecords(), 0.05);

        var newexample = new HashMap<String, String>();
        var header = records.getHeaderNames();
        for (i = 1; i < args.length; ++i) {
            newexample.put(header.get(i - 1), args[i]);
        }

        // optional chi-squared pruning
        tree.prune();
        
        tree.drawTree(new File(args[0]));
        
        boolean prediction = tree.predict(newexample);
        System.out.print("Prediction outcome: ");
        if (prediction == true) {
            System.out.println("+");
        } else {
            System.out.println("-");
        }
    }
}
