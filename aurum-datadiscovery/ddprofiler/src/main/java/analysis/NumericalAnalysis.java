/**
 *
 */
package analysis;

import analysis.modules.Range;
import sources.deprecated.Attribute.AttributeType;

/**
 * @author Raul - raulcf@csail.mit.edu
 * Sibo (edit)
 */
public interface NumericalAnalysis
    extends Analysis, IntegerDataConsumer, FloatDataConsumer {
  // add an interface to return the quantile
  public Range getNumericalRange(AttributeType type);
  public long getQuantile(double p);
  public boolean hasEnoughDataForQuantiles();
}

