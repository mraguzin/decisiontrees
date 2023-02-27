package com.mraguzin.decisiontrees;

/**
 *
 * @author mraguzin
 */
public class Pair <U, V> {
    U x;
    V y;
    
    public Pair(U x, V y) {
        this.x = x;
        this.y = y;
    }
    
    @Override
    public String toString() {
        return "(" + x.toString() + "," + y.toString() + ")";
    }
}
