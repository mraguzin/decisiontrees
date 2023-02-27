package com.mraguzin.decisiontrees;

import java.util.List;
import java.util.Random;
import org.apache.commons.csv.CSVRecord;

/**
 * A class containing miscellaneous helper functions.
 * @author mraguzin
 */
public class Helpers {
    private static final Random r = new Random();
    
    public static boolean majority(ClassDetector classifier, String classAttribute,
            List<CSVRecord> examples) {
        int nPluses = 0;
        int nMinuses = 0;
        
        for (var ex : examples) {
            if (classifier.classify(ex.get(classAttribute)) == true)
                ++nPluses;
            else
                ++nMinuses;
        }
        
        if (nPluses > nMinuses)
            return true;
        else if (nMinuses > nPluses)
            return false;
        else // random tie-break
            return r.nextDouble() < 0.5;
    }
    
    public static BoolPair sameClassification(ClassDetector classifier, String classAttribute,
            List<CSVRecord> examples) {
        boolean classification = false;
        boolean first = true;
        
        for (var ex : examples) {
            if (first) {
                classification = classifier.classify(ex.get(classAttribute));
                first = false;
            }
            else if (classifier.classify(ex.get(classAttribute)) != classification)
                return new BoolPair(false, false);
        }
        
        return new BoolPair(true, classification);
    }
    
    public static double getBinomialEntropy(double q) {
        if (q == 0 || q == 1)
            return 0;
        
        double oneMinus = 1 - q;
        double res = Math.log(q) / Math.log(2) * q + 
                oneMinus * Math.log(oneMinus) / Math.log(2);
        return -res; // TODO: optimise out the log(2) stuff?
    }
}
