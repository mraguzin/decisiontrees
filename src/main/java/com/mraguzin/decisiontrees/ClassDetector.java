package com.mraguzin.decisiontrees;

/**
 * Functional interface representing a boolean function on a string. Meant to be used in
 * form of a lambda function implementing a fast and simple filter for + and -
 * values on classification examples (a pattern matching '+' returns true; false
 * otherwise).
 * @author mraguzin
 */
@FunctionalInterface
public interface ClassDetector { boolean classify(String attributeName); }