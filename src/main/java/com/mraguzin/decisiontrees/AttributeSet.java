package com.mraguzin.decisiontrees;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * Set of attributes and their values. This class stores all the attributes and their
 * value sets corresponding to a single decision tree learning scenario.
 * @author mraguzin
 */
public class AttributeSet {
    private enum Type {CATEGORICAL, NUMERICAL}; // numeric atts. get discretised via
    // supervised binning and thus require special handling
    
    private HashMap<String, List<String>> stringValues = new HashMap<>();
    private HashMap<String, HashSet<String>> stringValueset = new HashMap<>();
    private HashMap<String, List<Pair<Double, Boolean>>> numericValues = new HashMap<>();
    private HashMap<String, List<Double>> discretisedNumericValues;
    private final int positives, negatives;
    private int inputCounter;
    private final int nAttributes;
    private final static int MAX_SPLITS = 10;
    private int attributeSplitCount;
    
    public AttributeSet(int nAttributes, int nPositives, int nNegatives) {
        this.nAttributes = nAttributes - 1; // ignore the class input attribute
        this.positives = nPositives;
        this.negatives = nNegatives;
    }
    
    /**
     * Associates the provided value list with the given attribute. This method puts
     * the provided list of (value,classification) pairs into their properly
     * typed list. It also takes care of all necessary conversion in case of
     * numeric inputs (it assumes that, if the first seen example is numeric,
     * then all the remaining entries will be as well) and finally, after all
     * known attributes have been processed, triggers the discretisation phase
     * in case the attribute is numerical.
     * @param attribute
     * @param values 
     */
    public void put(String attribute, List<Pair<String, Boolean>> values) {
        boolean isNumeric = true;
        
        ++inputCounter;
        
        try {
            Double.valueOf(values.get(0).x);
        } catch (NumberFormatException ex) {
            isNumeric = false;
        }
        
        if (isNumeric) {
            System.out.println("NUMERIC ATTRIBUTE");
            for (var value : values) {
                
                if (numericValues.get(attribute) == null) {
                List<Pair<Double, Boolean>> doubleList = new ArrayList<>();
                doubleList.add(new Pair<>(Double.valueOf(value.x), value.y));
                numericValues.put(attribute, doubleList);
            }
                else {
                    var doubleList = numericValues.get(attribute);
                    doubleList.add(new Pair<>(Double.valueOf(value.x), value.y));
                    numericValues.put(attribute, doubleList);
                }
            }
        }
        
        else {        
        for (var value : values) {            
            if (stringValueset.get(attribute) == null) {
                var stringSet = new HashSet<String>();
                stringSet.add(value.x);
                stringValueset.put(attribute, stringSet);
            }
            else {
                var stringSet = stringValueset.get(attribute);
                stringSet.add(value.x);
                stringValueset.put(attribute, stringSet);
            }
        }
        }
        
        System.out.println("natts=" + (stringValueset.size() + numericValues.size()));
        System.out.println(stringValues);
        
        if (inputCounter == nAttributes) { // trigger post-processing
            for (Map.Entry<String, HashSet<String>> current : stringValueset.entrySet()) {
                String name = current.getKey();
                var tmpList = new ArrayList();
                tmpList.addAll(current.getValue());
                stringValues.put(name, tmpList);                
            }
            discretiseAllNumericAttributes();
            
            //clean up useless collections
            stringValueset = null;
            numericValues = null;
        }
            
    }
    
    private boolean isNumeric(String attribute) {
        return discretisedNumericValues.containsKey(attribute);
    }
    
    /**
     * Checks whether the given attribute value matches the learned values.Specifically,
 for generic attributes this simply reduces to equality, but for numerical
 ones it necessitates a search, since we only store split point thresholds.
     * @param attribute The name of the input attribute
     * @param input First value to compare with, usually from user input
     * @param value Second value to compare with; supposed to be the learned datum
     * @return Whether the input matches the learned datum or not.
     */
    public boolean containedIn(String attribute, String input, Object value ) {
        if (isNumeric(attribute)) {
            double numInput = Double.parseDouble(input);
            // search for the right interval
            int inputBin = findBin(attribute, numInput);
            int valueBin = findBin(attribute, (Double)value);
            return inputBin == valueBin;            
        }
        else {
            // it's just a categorical attribute
            return input.equals(value.toString());
        }
    }
    
    /**
     * Produces a label string describing this attribute's value. Suitable for labeling
     * decision tree edges.
     * @param attribute
     * @param value
     * @return The label
     */
    public String getLabel(String attribute, Object value) {
        if (isNumeric(attribute)) {
            var df = DecimalFormat.getInstance();
            df.setMaximumFractionDigits(2);
            
            if (!(value instanceof Double))
                value = Double.valueOf(value.toString());
            int bin = findBin(attribute, (Double)value);
            var list = discretisedNumericValues.get(attribute);
            if (bin == list.size() - 1)
                return ">" + df.format(list.get(list.size()-2));
            else if (bin == 0)
                return "≤" + df.format(list.get(0));
            else
                return df.format(list.get(bin - 1)) + 
                        "-" + df.format(list.get(bin)); // an [x,y> range       
        }
        
        else
            return value.toString();
    }
    
    private int findBin(String attribute, double value) {
        var ranges = discretisedNumericValues.get(attribute);
        
        for (int i = 0; i < ranges.size(); ++i) {
            if (value <= ranges.get(i))
                return i;
        }
        
        return ranges.size() - 1; // outside the learned range
    }
    
    /**
     * Gets the total number of values for the given attribute.
     * @param attribute
     * @return Number of values attribute can take on.
     */
    public int size(String attribute) {
        List list = stringValues.get(attribute);
        if (list == null)
            list = discretisedNumericValues.get(attribute);
        if (list == null)
            return -1;
        
        return list.size();
    }
    
    /**
     * Randomly shuffles the provided attribute's value set. The idea here is to counter
     * the possibility of very deep trees arising due to an attribute taking on
     * many possible values. The threshold for using this is defined within the
     * decision tree learner itself; it proceeds by then only using a smaller
     * subset of shuffled values.
     * @param attribute 
     */
    public void shuffle(String attribute) {
        List<String> list = stringValues.get(attribute);
        if (list == null)
            return;
        
        Collections.shuffle(list);
    }
    
    public List get(String attribute) {
        List list = stringValues.get(attribute);
        if (list != null)
            return list;
        else
            return discretisedNumericValues.get(attribute);
    }
    
    public List getSublist(String attribute, int size) {
        List list = stringValues.get(attribute);
        if (list != null)
            return list.subList(0, size);
        else
            return discretisedNumericValues.get(attribute).subList(0, size);
    }
    
    private void discretiseAllNumericAttributes() {
        discretisedNumericValues = new HashMap<>();
        for (Map.Entry<String, List<Pair<Double, Boolean>>> entry : numericValues.entrySet()) {
            // pre-sort the values so that the algorithm below can actually work
            // (efficiently)
            entry.getValue().sort((a, b) -> a.x.compareTo(b.x));
            attributeSplitCount = 0;
            var splitPointList = bin(positives, negatives, entry.getValue());

            splitPointList.add(Double.POSITIVE_INFINITY); // to catch <last,∞>
            discretisedNumericValues.put(entry.getKey(), splitPointList);
            System.out.println(splitPointList);
            System.out.println("Done with " + entry.getKey());
        }
    }
    
    /**
     * Transforms the numerical attribute into a categorical one. It does so by a recursive
     * partitioning process, maximizing information gain as it goes along the
     * sorted value set. This algorithm is briefly mentioned in Russell&Norvig
     * 3rd ed. p720 and sketched <a href=https://www.saedsayad.com/supervised_binning.htm>here</a>.
     * @param nPositive Number of positive examples in the input list
     * @param nNegative Number of negative examples in the input list
     * @param values A list of (value,classification) pairs for a single attribute
     * @return The list of "optimal" split points.
     */
    private List<Double> bin(int nPositive, int nNegative, List<Pair<Double, Boolean>> values) {
        if (attributeSplitCount > MAX_SPLITS)
            return List.of();
        if (values.isEmpty() || nNegative + nPositive == 0)
            return List.of();
        else if (values.size() == 1)
            return List.of();
            //return List.of(values.get(0).x);
        
        System.out.println("p=" + nPositive + ", n=" + nNegative);
        System.out.println(values);
        
        double goalEntropy = Helpers.getBooleanEntropy((double)nPositive / (nPositive + nNegative));
        boolean lastChange = values.get(0).y;
        double maxGain = Double.MIN_VALUE;
        double splitThreshold = Double.NaN;
        int pLeft = lastChange ? 1 : 0;
        int nLeft = lastChange ? 0 : 1;
        int sampleSize = nPositive + nNegative;
        List<Double> splitList = new ArrayList<>(); // this is filled up
        // recursively and returned at the end
        int bestLb = 0;
        int bestRb = 0;
        int pLeftBest = pLeft;
        int nLeftBest = nLeft;
        
        for (int i = 1; i < values.size(); ++i) {
            var pair = values.get(i);
            if (pair.y == true)
                ++pLeft;
            else
                ++nLeft;
            
            if (lastChange != pair.y) {
                Double x = pair.x;
                double mean = (values.get(i - 1).x + pair.x) * 0.5;
                //double mean = pair.x;
                int lb = i;
                int j;
                for (j = i+1; j < values.size() &&
                        values.get(j).x.equals(x); ++j) {                    
                    if (values.get(j).y == true)
                        ++pLeft;
                    else
                        ++nLeft;
                    
                    lastChange = values.get(j).y;
                }
                
                i = j - 1;
                int rb = j;
                int pRight = nPositive - pLeft;
                int nRight = nNegative - nLeft;
                
                double entropy = 0;
                // left side (<=)
                double p1 = (double)pLeft / (pLeft + nLeft);
                entropy += (double)(pLeft + nLeft) / sampleSize * Helpers.getBooleanEntropy(p1);
                // right side (>)
                double p2 = (double)pRight / (pRight + nRight);
                entropy += (double)(pRight + nRight) / sampleSize * Helpers.getBooleanEntropy(p2);
                
                double gain = goalEntropy - entropy;
                if (gain > maxGain) {
                    maxGain = gain;
                    splitThreshold = mean;
                    bestLb = lb;
                    bestRb = rb;
                    pLeftBest = pLeft;
                    nLeftBest = nLeft;
                }
            }
        }
        
        if (Double.isNaN(splitThreshold))
            return List.of();
        
        // recurse
        int toTheLeft = bestLb;
        int toTheRight = bestRb;
        System.out.println("lb="+bestLb +", rb=" + bestRb);
        // find the upper bound, because the right side must be > and there
        // might be a whole subsequence of duplicate values
        List<Double> leftValues;
        List<Double> rightValues;
        try {
            leftValues = bin(pLeftBest, nLeftBest, values.subList(0, toTheLeft));
        } catch (IllegalArgumentException e) {
            leftValues = List.of();
        }
        
        try {
            rightValues = bin(nPositive - pLeftBest, nNegative - nLeftBest,
                    values.subList(toTheRight, values.size()));
        } catch (IllegalArgumentException e) {
            rightValues = List.of();
        }
              
        // merge, keeping everything in sorted order
        splitList.addAll(leftValues);
        splitList.add(splitThreshold);
        splitList.addAll(rightValues);
        ++attributeSplitCount;
        
        return splitList;
    }
}
