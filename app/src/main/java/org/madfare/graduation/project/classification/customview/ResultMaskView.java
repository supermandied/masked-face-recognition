package org.tensorflow.lite.examples.classification.customview;

import java.util.List;
import org.tensorflow.lite.examples.classification.tflite.MaskClassifier.Recognition;

public interface ResultMaskView {
    public void setResults(final List<Recognition> results);
}
