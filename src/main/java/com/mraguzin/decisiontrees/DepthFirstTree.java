package com.mraguzin.decisiontrees;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;

/**
 *
 * This class represents a decision tree built in a depth-first manner.
 *
 * @author mraguzin
 */
public class DepthFirstTree {

    private Map<String, Integer> attributeNames;
    //private int classAttributeIndex;
    private String classAttribute;
    private ClassDetector classifier;
    private Map<String, DepthFirstTree> children; // the string holds the
    // particular branch label
    private String rootAttribute; // this doubles as the classification string
    // (Yes/No or similar) for leaf nodes
    private AttributeSet attributeValues; // holds the set
    // of all values a particular attribute can take on
    private List<CSVRecord> allExamples; // all examples relevant for this particular (sub)tree
    private double statisticalSignificance;

    private static final Random r = new Random();
    private static final int MULTIVALUE_THRESHOLD = 10; // threshold on the number
    // of different attribute values which, when crossed, causes the greedy
    // heuristic to only pick from a smaller, randomly selected subset of
    // values (of size multivalueThreshold)

    private DepthFirstTree(Map<String, Integer> attributes, String classAttribute,
            ClassDetector classifier, List<CSVRecord> allExamples,
            AttributeSet attributeValues,
            List<CSVRecord> parentExamples, double pvalue) {
        attributeNames = attributes;
        this.classAttribute = classAttribute;
        this.classifier = classifier;
        this.allExamples = allExamples;
        statisticalSignificance = pvalue;
        children = new HashMap<>();
        System.out.println("numexamples=" + allExamples.size());

        if (attributeValues == null) {
            getAttributeValues(allExamples, attributes); // we need all the training data up-front
            // so as to be able to construct the attribute-value sets a priori
            // (a possible optimisation?)
        } else {
            this.attributeValues = attributeValues;
        }

        buildDecisionTree(allExamples, parentExamples);
    }

    public DepthFirstTree(Map<String, Integer> attributes, String classAttribute,
            ClassDetector classifier, List<CSVRecord> allExamples, double pvalueCutoff) {
        this(attributes, classAttribute, classifier, allExamples, null, null, pvalueCutoff);
    }

    private void getAttributeValues(List<CSVRecord> examples, Map<String, Integer> attributes) {
        int p = 0;
        int n = 0;
        for (var ex : examples) {
            if (classifier.classify(ex.get(classAttribute)) == true) {
                ++p;
            } else {
                ++n;
            }
        }

        attributeValues = new AttributeSet(attributes.size(), p, n);

        for (Map.Entry<String, Integer> entry : attributeNames.entrySet()) {
            String attribute = entry.getKey();
            if (attribute.equals(classAttribute)) {
                continue;
            }

            var list = new ArrayList<Pair<String, Boolean>>();
            for (var ex : examples) {
                boolean test = classifier.classify(ex.get(classAttribute));
                list.add(new Pair<>(ex.get(attribute), test));
            }

            attributeValues.put(attribute, list);
        }
    }

    /**
     * Predicts + or - on the basis of the provided (attribute,value) pairs,
     * given as a map.
     *
     * @param attributes
     * @return Whether the input example classifies as + (true) or - (false).
     */
    public boolean predict(Map<String, String> attributes) {
        if (attributes.size() != attributeNames.size() - 1) {
            throw new IllegalArgumentException("Length of example record has "
                    + "to be equal to the number of training attributes");
        }

        return _predict(attributes);
    }

    private boolean _predict(Map<String, String> attributes) {
        if (children.isEmpty()) {
            return rootAttribute.equals("+");
        }

        String valueLabel = attributeValues
                .getLabel(rootAttribute, attributes.get(rootAttribute));

        return children.get(valueLabel)._predict(attributes);
    }

    private void buildDecisionTree(List<CSVRecord> examples,
            List<CSVRecord> parentExamples) {
        BoolPair examplesClass;
        var attributes = attributeNames;

        if (examples.isEmpty()) {
            boolean majority = Helpers.majority(classifier,
                    classAttribute, parentExamples);
            if (majority == true) {
                rootAttribute = "+";
            } else {
                rootAttribute = "-";
            }
        } else if ((examplesClass = Helpers.sameClassification(classifier, classAttribute, examples)).x) {
            if (examplesClass.y == true) {
                rootAttribute = "+";
            } else {
                rootAttribute = "-";
            }
        } else if (attributes.isEmpty()) {
            boolean majority = Helpers.majority(classifier, classAttribute, examples);
            if (majority == true) {
                rootAttribute = "+";
            } else {
                rootAttribute = "-";
            }
        } else {
            System.out.println("atnames_size=" + attributeNames.size());
            System.out.println("examples_size=" + examples.size());
            var it = attributeNames.entrySet().iterator();
            String maxAttribute = "";
            double maxImportance = Double.MIN_VALUE;

            while (it.hasNext()) { // employ the greedy heuristic and attempt
                // to only work on a smaller subset if we've crossed the size threshold
                var attribute = it.next();
                if (attribute.getKey().equals(classAttribute)) {
                    continue;
                }
                double importance = getImportance(attribute.getKey(), examples);
                System.out.println(importance);
                if (importance > maxImportance) {
                    maxImportance = importance;
                    maxAttribute = attribute.getKey();
                }
            }

            rootAttribute = maxAttribute;

            System.out.println("maxat=" + maxAttribute);
            var valueList = attributeValues.get(maxAttribute);
            var subAttributeNames = new HashMap<String, Integer>();
            subAttributeNames.putAll(attributeNames);
            subAttributeNames.remove(maxAttribute);

            for (var value : valueList) {
                var subExamples = new ArrayList<CSVRecord>();
                for (var ex : examples) {
                    if (attributeValues.containedIn(maxAttribute, ex.get(maxAttribute), value)) {
                        subExamples.add(ex);
                    }
                }

                var subtree = new DepthFirstTree(subAttributeNames,
                        classAttribute, classifier, subExamples, attributeValues,
                        examples, statisticalSignificance);

                String valueLabel = attributeValues.getLabel(maxAttribute, value);
                children.put(valueLabel, subtree); // recurse
            }
        }
    }

    /**
     * Prunes the built decision tree using a chi-squared distribution test on
     * every node with only leaf nodes as children.
     *
     * @return Whether this subtree was pruned.
     */
    public boolean prune() {
        if (children.isEmpty()) {
            return true;
        }

        boolean childrenPruned = true; // whether all the children were pruned
        // or were already leaves
        for (var child : children.values()) {
            childrenPruned &= child.prune();
        }

        if (childrenPruned) {
            // attempt to prune this node
            var pair = getExampleDistribution(allExamples);
            int p = pair.x;
            int n = pair.y;
            double delta = 0;
            int degreesOfFreedom = attributeValues.size(rootAttribute) - 1;

            for (var val : attributeValues.get(rootAttribute)) {
                pair = getExampleDistributionWithValue(allExamples, rootAttribute, val);
                int pk = pair.x;
                int nk = pair.y;
                double pkExpected = p * (double) (pk + nk) / (p + n);
                double nkExpected = n * (double) (pk + nk) / (p + n);

                delta += (pk - pkExpected) * (pk - pkExpected) / pkExpected
                        + (nk - nkExpected) * (nk - nkExpected) / nkExpected;
            }

            double alpha = 1 - statisticalSignificance;
            var chiSquared = new ChiSquaredDistribution(degreesOfFreedom);
            double quantile = chiSquared.inverseCumulativeProbability(alpha);
            if (delta < quantile) {
                // prune                
                int nPluses = 0;
                int nMinuses = 0;
                for (var child : children.values()) {
                    if ("+".equals(child.rootAttribute)) {
                        ++nPluses;
                    } else {
                        ++nMinuses;
                    }
                }

                String majority;
                if (nPluses > nMinuses) {
                    majority = "+";
                } else if (nMinuses > nPluses) {
                    majority = "-";
                } else {
                    majority = r.nextDouble() < 0.5 ? "+" : "-";
                }

                children.clear();
                rootAttribute = majority;
                return true;
            }
        }

        return false;
    }

    /**
     * Returns the number of positive and negative examples in the given list.
     *
     * @param examples
     * @return A (positive,negative) count pair.
     */
    public Pair<Integer, Integer> getExampleDistribution(List<CSVRecord> examples) {
        return _getExampleDistribution(examples, null, null);
    }

    /**
     * Returns the number of positive and negative examples for which
     * attribute=value holds.
     *
     * @param examples
     * @param attribute
     * @param value
     * @return A (positive,negative) count pair.
     */
    public Pair<Integer, Integer> getExampleDistributionWithValue(List<CSVRecord> examples,
            String attribute, Object value) {
        return _getExampleDistribution(examples, attribute, value);
    }

    private Pair<Integer, Integer> _getExampleDistribution(List<CSVRecord> examples,
            String attribute, Object value) {
        int p = 0;
        int n = 0;

        for (var ex : examples) {
            if (value == null || attributeValues.containedIn(attribute, ex.get(attribute), value)) {
                if (classifier.classify(ex.get(classAttribute)) == true) {
                    ++p;
                } else {
                    ++n;
                }
            }
        }

        return new Pair<>(p, n);
    }

    /**
     * Compute information gain for the given attribute. This is to be used in
     * the greedy decision tree builder as a value indicating the importance of
     * an attribute.
     *
     * @param attribute
     * @param examples
     * @return Gain
     */
    private double getImportance(String attribute, List<CSVRecord> examples) {
        var pair = getExampleDistribution(examples);
        int p = pair.x;
        int n = pair.y;

        // compute Remainder(attribute)
        double remainder = 0;

        int size = attributeValues.size(attribute);
        List valueSubset;
        if (size > MULTIVALUE_THRESHOLD) {
            attributeValues.shuffle(attribute);
            valueSubset = attributeValues.getSublist(attribute, MULTIVALUE_THRESHOLD);
        } else {
            valueSubset = attributeValues.get(attribute);
        }

        for (var val : valueSubset) {
            var tmp = getExampleDistributionWithValue(examples, attribute, val);
            int pk = tmp.x;
            int nk = tmp.y;

            double q = pk / (double) (pk + nk);
            if (Double.isNaN(q)) {
                continue;
            }

            remainder += ((pk + nk) / (double) (p + n)) * Helpers.getBinomialEntropy(q);
        }

        double q = p / (double) (p + n);
        return Helpers.getBinomialEntropy(q) - remainder;
    }
}
