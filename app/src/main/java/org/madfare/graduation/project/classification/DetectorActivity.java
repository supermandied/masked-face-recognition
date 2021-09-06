package org.tensorflow.lite.examples.classification;


import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.camera2.CameraCharacteristics;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import org.tensorflow.lite.examples.classification.customview.OverlayView;
import org.tensorflow.lite.examples.classification.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.classification.env.BorderedText;
import org.tensorflow.lite.examples.classification.env.ImageUtils;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier;
import org.tensorflow.lite.examples.classification.tflite.MaskClassifier;
import org.tensorflow.lite.examples.classification.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.classification.tracking.MultiBoxTracker;

/**
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged SSD model.
    //private static final int TF_OD_API_INPUT_SIZE = 300;
    //private static final boolean TF_OD_API_IS_QUANTIZED = true;
    //private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    //private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";

    //private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    // Face Mask
    private static final int TF_OD_API_INPUT_SIZE_MASK = 224;
    private static final boolean TF_OD_API_IS_QUANTIZED_MASK = false;
    private static final String TF_OD_API_MODEL_FILE_MASK = "maskDetection3.tflite";
    private static final String TF_OD_API_LABELS_FILE_MASK = "file:///android_asset/Alabels.txt";

    private static final DetectorMode MODE_MASK = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API_MASK = 0.5f;
    private static final boolean MAINTAIN_ASPECT_MASK = false;

    private static final Size DESIRED_PREVIEW_SIZE_MASK = new Size(800, 600);
    //private static final int CROP_SIZE = 320;
    //private static final Size CROP_SIZE = new Size(320, 320);



    private static final boolean SAVE_PREVIEW_BITMAP_MASK = false;
    private static final float TEXT_SIZE_DIP_MASK = 10;
    OverlayView trackingOverlay_MASK;
    private Integer sensorOrientation_MASK;

    private MaskClassifier detector_MASK;

    private long lastProcessingTimeMs_MASK;
    private Bitmap rgbFrameBitmap_MASK = null;
    private Bitmap croppedBitmap_MASK = null;
    private Bitmap cropCopyBitmap_MASK = null;

    private boolean computingDetection_MASK = false;

    private long timestamp_MASK = 0;

    private Matrix frameToCropTransform_MASK;
    private Matrix cropToFrameTransform_MASK;

    private MultiBoxTracker tracker_MASK;

    private BorderedText borderedText_MASK;

    // Face detector
    private FaceDetector faceDetector_MASK;

    // here the preview image is drawn in portrait way
    private Bitmap portraitBmp_MASK = null;
    // here the face is cropped and drawn
    private Bitmap faceBmp_MASK = null;

    //ClassifierActivity

    private static final boolean MAINTAIN_ASPECT = true;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final float TEXT_SIZE_DIP = 10;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;
    private long lastProcessingTimeMs;
    private Integer sensorOrientation;
    private Classifier classifier;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private BorderedText borderedText;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        // Real-time contour detection of multiple faces
        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                        .setContourMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                        .build();


        FaceDetector detector = FaceDetection.getClient(options);

        faceDetector_MASK = detector;


        //checkWritePermission();

    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx_MASK =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP_MASK, getResources().getDisplayMetrics());
        borderedText_MASK = new BorderedText(textSizePx_MASK);
        borderedText_MASK.setTypeface(Typeface.MONOSPACE);

        tracker_MASK = new MultiBoxTracker(this);


        try {
            detector_MASK =
                    TFLiteObjectDetectionAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE_MASK,
                            TF_OD_API_LABELS_FILE_MASK,
                            TF_OD_API_INPUT_SIZE_MASK,
                            TF_OD_API_IS_QUANTIZED_MASK);
            //cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        int screenOrientation = getScreenOrientation();
        sensorOrientation_MASK = rotation - screenOrientation;
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation_MASK);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap_MASK = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);




        int targetW, targetH;
        if (sensorOrientation_MASK == 90 || sensorOrientation_MASK == 270) {
            targetH = previewWidth;
            targetW = previewHeight;
        }
        else {
            targetW = previewWidth;
            targetH = previewHeight;
        }
        int cropW = (int) (targetW / 2.0);
        int cropH = (int) (targetH / 2.0);

        croppedBitmap_MASK = Bitmap.createBitmap(cropW, cropH, Config.ARGB_8888);

        portraitBmp_MASK = Bitmap.createBitmap(targetW, targetH, Config.ARGB_8888);
        faceBmp_MASK = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE_MASK, TF_OD_API_INPUT_SIZE_MASK, Config.ARGB_8888);

        frameToCropTransform_MASK =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropW, cropH,
                        sensorOrientation_MASK, MAINTAIN_ASPECT_MASK);

//    frameToCropTransform =
//            ImageUtils.getTransformationMatrix(
//                    previewWidth, previewHeight,
//                    previewWidth, previewHeight,
//                    sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform_MASK = new Matrix();
        frameToCropTransform_MASK.invert(cropToFrameTransform_MASK);



        trackingOverlay_MASK = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay_MASK.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker_MASK.draw(canvas);

                    }
                });

        tracker_MASK.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation_MASK);
    }


    @Override
    protected void processImage() {
        ++timestamp_MASK;
        final long currTimestamp = timestamp_MASK;
        trackingOverlay_MASK.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection_MASK) {
            readyForNextImage();
            return;
        }
        computingDetection_MASK = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap_MASK.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap_MASK);
        canvas.drawBitmap(rgbFrameBitmap_MASK, frameToCropTransform_MASK, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP_MASK) {
            ImageUtils.saveBitmap(croppedBitmap_MASK);
        }

        InputImage image = InputImage.fromBitmap(croppedBitmap_MASK, 0);
        faceDetector_MASK
                .process(image)
                .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                    @Override
                    public void onSuccess(List<Face> faces) {
                        if (faces.size() == 0) {
                            updateResults(currTimestamp, new LinkedList<>());
                            return;
                        }
                        runInBackground(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        onFacesDetected(currTimestamp, faces);
                                    }
                                });
                    }

                });


    }

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE_MASK;
    }

    @Override
    protected void onInferenceConfigurationChanged() {

    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }


    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector_MASK.setUseNNAPI(isChecked));
    }


    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector_MASK.setNumThreads(numThreads));
    }


    // Face Mask Processing
    private Matrix createTransform(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation) {

        Matrix matrix = new Matrix();
        if (applyRotation != 0) {
            if (applyRotation % 90 != 0) {
                LOGGER.w("Rotation of %d % 90 != 0", applyRotation);
            }

            // Translate so center of image is at origin.
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

            // Rotate around origin.
            matrix.postRotate(applyRotation);
        }

//        // Account for the already applied rotation, if any, and then determine how
//        // much scaling is needed for each axis.
//        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;
//
//        final int inWidth = transpose ? srcHeight : srcWidth;
//        final int inHeight = transpose ? srcWidth : srcHeight;

        if (applyRotation != 0) {

            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }

        return matrix;

    }


    private void updateResults(long currTimestamp, final List<MaskClassifier.Recognition> mappedRecognitions) {

        tracker_MASK.trackResults(mappedRecognitions, currTimestamp);
        trackingOverlay_MASK.postInvalidate();
        computingDetection_MASK = false;


        runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        showFrameInfo(previewWidth + "x" + previewHeight);
                        showCropInfo(croppedBitmap_MASK.getWidth() + "x" + croppedBitmap_MASK.getHeight());
                        showInference(lastProcessingTimeMs_MASK + "ms");
                    }
                });

    }

    private void onFacesDetected(long currTimestamp, List<Face> faces) {

        cropCopyBitmap_MASK = Bitmap.createBitmap(croppedBitmap_MASK);
        final Canvas canvas = new Canvas(cropCopyBitmap_MASK);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Style.STROKE);
        paint.setStrokeWidth(2.0f);

        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API_MASK;
        switch (MODE_MASK) {
            case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API_MASK;
                break;
        }

        final List<MaskClassifier.Recognition> mappedRecognitions =
                new LinkedList<MaskClassifier.Recognition>();


        //final List<Classifier.Recognition> results = new ArrayList<>();

        // Note this can be done only once
        int sourceW = rgbFrameBitmap_MASK.getWidth();
        int sourceH = rgbFrameBitmap_MASK.getHeight();
        int targetW = portraitBmp_MASK.getWidth();
        int targetH = portraitBmp_MASK.getHeight();
        Matrix transform = createTransform(
                sourceW,
                sourceH,
                targetW,
                targetH,
                sensorOrientation_MASK);
        final Canvas cv = new Canvas(portraitBmp_MASK);

        // draws the original image in portrait mode.
        cv.drawBitmap(rgbFrameBitmap_MASK, transform, null);

        final Canvas cvFace = new Canvas(faceBmp_MASK);

        boolean saved = false;

        for (Face face : faces) {

            LOGGER.i("FACE" + face.toString());

            LOGGER.i("Running detection on face " + currTimestamp);

            //results = detector.recognizeImage(croppedBitmap);


            final RectF boundingBox = new RectF(face.getBoundingBox());

            //final boolean goodConfidence = result.getConfidence() >= minimumConfidence;
            final boolean goodConfidence = true; //face.get;
            if (boundingBox != null && goodConfidence) {

                // maps crop coordinates to original
                cropToFrameTransform_MASK.mapRect(boundingBox);

                // maps original coordinates to portrait coordinates
                RectF faceBB = new RectF(boundingBox);
                transform.mapRect(faceBB);

                // translates portrait to origin and scales to fit input inference size
                //cv.drawRect(faceBB, paint);
                float sx = ((float) TF_OD_API_INPUT_SIZE_MASK) / faceBB.width();
                float sy = ((float) TF_OD_API_INPUT_SIZE_MASK) / faceBB.height();
                Matrix matrix = new Matrix();
                matrix.postTranslate(-faceBB.left, -faceBB.top);
                matrix.postScale(sx, sy);

                cvFace.drawBitmap(portraitBmp_MASK, matrix, null);


                String label = "";
                float confidence = -1f;
                Integer color = Color.BLUE;

                final long startTime = SystemClock.uptimeMillis();
                final List<MaskClassifier.Recognition> resultsAux = detector_MASK.recognizeImage(faceBmp_MASK);
                lastProcessingTimeMs_MASK = SystemClock.uptimeMillis() - startTime;

                if (resultsAux.size() > 0) {

                    MaskClassifier.Recognition result = resultsAux.get(0);

                    float conf = result.getConfidence();
                    if (conf >= 0.6f) {

                        confidence = conf;
                        label = result.getTitle();
                        if (result.getId().equals("0")) {
                            color = Color.GREEN;
                        }
                        else if(result.getId().equals("2")){
                            color = Color.YELLOW;
                        }
                        else {
                            color = Color.RED;
                        }
                    }

                }



                final MaskClassifier.Recognition result = new MaskClassifier.Recognition(
                        "0", label, confidence, boundingBox);

                result.setColor(color);
                result.setLocation(boundingBox);
                mappedRecognitions.add(result);


            }


        }

        //    if (saved) {
//      lastSaved = System.currentTimeMillis();
//    }

        updateResults(currTimestamp, mappedRecognitions);


    }


}
