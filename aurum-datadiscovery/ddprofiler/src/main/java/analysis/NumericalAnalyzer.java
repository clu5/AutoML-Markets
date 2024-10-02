/**
 * @author Raul - raulcf@csail.mit.edu
 *
 */
package analysis;

import java.util.ArrayList;
import java.util.List;

import analysis.modules.Cardinality;
import analysis.modules.CardinalityAnalyzer;
import analysis.modules.Range;
import analysis.modules.RangeAnalyzer;
import sources.deprecated.Attribute.AttributeType;

public class NumericalAnalyzer implements NumericalAnalysis {

  private List<DataConsumer> analyzers;
  private CardinalityAnalyzer ca;
  private RangeAnalyzer ra;

  private NumericalAnalyzer() {
    analyzers = new ArrayList<>();
    ca = new CardinalityAnalyzer();
    ra = new RangeAnalyzer();
    analyzers.add(ca);
    analyzers.add(ra);
  }

  public static NumericalAnalyzer makeAnalyzer() {
    return new NumericalAnalyzer();
  }

  @Override
  public boolean feedIntegerData(List<Long> records) {
    for (DataConsumer dc : analyzers) {
      if (dc instanceof IntegerDataConsumer) {
        ((IntegerDataConsumer) dc).feedIntegerData(records);
      }
    }
    return true;
  }

  @Override
  public boolean feedFloatData(List<Float> records) {
    for (DataConsumer dc : analyzers) {
      if (dc instanceof FloatDataConsumer) {
        ((FloatDataConsumer) dc).feedFloatData(records);
      }
    }
    return true;
  }

  @Override
  public DataProfile getProfile() {
    // TODO: Implement this method
    return null;
  }

  @Override
  public Cardinality getCardinality() {
    return ca.getCardinality();
  }

  @Override
  public Range getNumericalRange(AttributeType type) {
    if (type.equals(AttributeType.FLOAT)) {
      return ra.getFloatRange();
    } else if (type.equals(AttributeType.INT)) {
      return ra.getIntegerRange();
    }
    return null;
  }

  @Override
  public long getQuantile(double p) {
    return ra.getQuantile(p);
  }

  @Override
  public boolean hasEnoughDataForQuantiles() {
    // Implement this method based on your requirements
    // For example, you might want to check if you have at least 30 data points
    return ca.getCardinality().getTotalRecords() >= 30;
  }
}

