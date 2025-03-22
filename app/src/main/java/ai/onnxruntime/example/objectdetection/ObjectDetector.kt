package ai.onnxruntime.example.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import androidx.core.graphics.scale
import kotlin.div

data class Result(
    var outputBitmap: Bitmap,
    var outputBox: Array<FloatArray>
)

internal class ObjectDetector {

    companion object {
        const val TAG = "ObjectDetector"
        const val INPUT_SIZE = 640
        const val NUM_THREADS = 4
        const val MODEL_G1 = "g1_best_float32.tflite"
        const val MODEL_G2 = "g2_best_float32.tflite"
        const val BOTH_MODELS = "both_models"

        private var scaleFactor: Float = 1.0f
        private var offsetX: Int = 0
        private var offsetY: Int = 0

        fun getScaleFactor(): Float = scaleFactor
        fun getOffsetX(): Int = offsetX
        fun getOffsetY(): Int = offsetY

        private var instance: ObjectDetector? = null
        private var lastPreprocessedBitmap: Bitmap? = null

        fun getInstance(): ObjectDetector {
            if (instance == null) {
                instance = ObjectDetector()
            }
            return instance!!
        }

        fun getPreprocessedInputBitmap(): Bitmap? {
            return lastPreprocessedBitmap
        }
    }

    private var tfliteG1: Interpreter? = null
    private var tfliteG2: Interpreter? = null
    private var currentModel: String = MODEL_G1
    private val bothModelsMerger = BothModelsMerger()



    fun initialize(context: Context, modelName: String = MODEL_G1) {

        currentModel = modelName
        if (modelName == MODEL_G1 || modelName == BOTH_MODELS) {
            initializeModel(context, MODEL_G1)
        }

        if (modelName == MODEL_G2 || modelName == BOTH_MODELS) {
            initializeModel(context, MODEL_G2)
        }
    }

    private fun initializeModel(context: Context, modelName: String) {
        val tfliteOptions = Interpreter.Options().apply {
            setNumThreads(NUM_THREADS)
        }

        try {
            val modelFile = context.assets.openFd(modelName)
            val fileChannel = FileInputStream(modelFile.fileDescriptor).channel
            val modelBuffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                modelFile.startOffset,
                modelFile.declaredLength
            )

            if (modelName == MODEL_G1) {
                tfliteG1 = Interpreter(modelBuffer, tfliteOptions)
            } else {
                tfliteG2 = Interpreter(modelBuffer, tfliteOptions)
            }

            Log.d(TAG, "TFLite model $modelName loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading TFLite model $modelName", e)
            throw RuntimeException("Error loading TFLite model $modelName", e)
        }
    }

    fun detect(
        inputStream: InputStream,
        context: Context,
        confidenceThreshold: Float = 0.25f,
        modelName: String = MODEL_G1
    ): Result {

        if ((tfliteG1 == null && (modelName == MODEL_G1 || modelName == BOTH_MODELS)) ||
            (tfliteG2 == null && (modelName == MODEL_G2 || modelName == BOTH_MODELS)) ||
            currentModel != modelName) {
            initialize(context, modelName)
        }

        // Load and preprocess the image
        val originalBitmap = BitmapFactory.decodeStream(inputStream)

        // Prepare input and output buffers
        val inputBuffer = preprocessImage(originalBitmap)

        return when (modelName) {
            BOTH_MODELS -> {
                // Run both models and merge results
                val g1Result = runModel(MODEL_G1, inputBuffer, originalBitmap, confidenceThreshold)
                val g2Result = runModel(MODEL_G2, inputBuffer, originalBitmap, confidenceThreshold)

                // Merge results from both models
                mergeResults(g1Result, g2Result, originalBitmap, confidenceThreshold)
            }
            else -> {
                // Run single model
                runModel(modelName, inputBuffer, originalBitmap, confidenceThreshold)
            }
        }
    }

    private fun runModel(
        modelName: String,
        inputBuffer: ByteBuffer,
        bitmap: Bitmap,
        confidenceThreshold: Float
    ): Result {
        val outputShape = if (modelName == MODEL_G1) 71 else 93
        val outputBuffer = Array(1) { Array(outputShape) { FloatArray(8400) } }

        // Select which model to use
        val interpreter = if (modelName == MODEL_G1) tfliteG1 else tfliteG2

        // Run inference
        interpreter?.run(inputBuffer, outputBuffer)

        // Process results
        val detections = postprocessYOLOv8(
            outputBuffer[0],
            bitmap.width,
            bitmap.height,
            confidenceThreshold
        )

        return Result(bitmap, detections)
    }

    private fun mergeResults(
        g1Result: Result,
        g2Result: Result,
        bitmap: Bitmap,
        confidenceThreshold: Float
    ): Result {
        return bothModelsMerger.mergeResults(g1Result, g2Result, bitmap)
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Create a blank canvas with padding
        val paddedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(paddedBitmap)
        canvas.drawColor(Color.BLACK) // Fill with black background

        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        // Calculate scaling factor to fit within 640x640
        scaleFactor = minOf(
            INPUT_SIZE.toFloat() / originalWidth,
            INPUT_SIZE.toFloat() / originalHeight
        )

        // Calculate new dimensions
        val scaledWidth = (originalWidth * scaleFactor).toInt()
        val scaledHeight = (originalHeight * scaleFactor).toInt()

        offsetX = (INPUT_SIZE - scaledWidth) / 2
        offsetY = (INPUT_SIZE - scaledHeight) / 2

        // Resize and center the image on the canvas
        val resizedBitmap = bitmap.scale(scaledWidth, scaledHeight)
        canvas.drawBitmap(resizedBitmap, offsetX.toFloat(), offsetY.toFloat(), null)

        // Save the padded input image for visualization
        Companion.lastPreprocessedBitmap = paddedBitmap.copy(Bitmap.Config.ARGB_8888, false)

        // Extract pixels from the padded bitmap
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        paddedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        inputBuffer.rewind()
        for (pixel in pixels) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        return inputBuffer
    }


    private fun postprocessYOLOv8(
        outputBuffer: Array<FloatArray>,
        originalWidth: Int,
        originalHeight: Int,
        confidenceThreshold: Float
    ): Array<FloatArray> {
        val rawResults = mutableListOf<FloatArray>()
        val outputShape = outputBuffer.size

        // Process 8400 detection boxes
        for (i in 0 until 8400) {
            var maxClassScore = 0f
            var bestClassIdx = 0

            for (c in 4 until outputShape) {
                val score = outputBuffer[c][i]
                if (score > maxClassScore) {
                    maxClassScore = score
                    bestClassIdx = c - 4
                }
            }

            if (maxClassScore >= confidenceThreshold) {
                // Get normalized coordinates (0-1)
                val normX = outputBuffer[0][i]
                val normY = outputBuffer[1][i]
                val normW = outputBuffer[2][i]
                val normH = outputBuffer[3][i]

                // Convert to model space pixels (640x640)
                val modelX = normX * INPUT_SIZE
                val modelY = normY * INPUT_SIZE
                val modelW = normW * INPUT_SIZE
                val modelH = normH * INPUT_SIZE

                // Store both model coordinates and original image coordinates
                // Format: [originalX, originalY, originalW, originalH, confidence, classId, modelX, modelY, modelW, modelH]
                rawResults.add(floatArrayOf(
                    (modelX - offsetX) / scaleFactor,  // originalX
                    (modelY - offsetY) / scaleFactor,  // originalY
                    modelW / scaleFactor,              // originalW
                    modelH / scaleFactor,              // originalH
                    maxClassScore,                     // confidence
                    bestClassIdx.toFloat(),            // classId
                    modelX,                            // modelX - store for debug visualization
                    modelY,                            // modelY
                    modelW,                            // modelW
                    modelH                             // modelH
                ))
            }
        }

        return applyNMS(rawResults.toTypedArray(), 0.5f).toTypedArray()
    }

    // Non-Maximum Suppression to filter overlapping boxes
    private fun applyNMS(boxes: Array<FloatArray>, iouThreshold: Float): List<FloatArray> {

        if (boxes.isEmpty()) return emptyList()

        // Sort boxes by confidence (highest first)
        val sortedBoxes = boxes.sortedByDescending { it[4] }
        val selected = mutableListOf<FloatArray>()
        val active = BooleanArray(sortedBoxes.size) { true }

        // Process boxes in order of confidence
        for (i in sortedBoxes.indices) {
            // Skip if this box was already eliminated
            if (!active[i]) continue

            // Add this box to the selected list
            selected.add(sortedBoxes[i])

            // Compare with remaining boxes
            for (j in i + 1 until sortedBoxes.size) {
                // Skip already eliminated boxes
                if (!active[j]) continue

                // If IOU is high enough, eliminate box j
                if (calculateIOU(sortedBoxes[i], sortedBoxes[j]) >= iouThreshold) {
                    active[j] = false
                }
            }
        }

        return selected
    }

    // Calculate Intersection over Union between two boxes
    private fun calculateIOU(box1: FloatArray, box2: FloatArray): Float {
        // Convert from center format (x,y,w,h) to corner format (x1,y1,x2,y2)
        val box1X1 = box1[0] - box1[2] / 2
        val box1Y1 = box1[1] - box1[3] / 2
        val box1X2 = box1[0] + box1[2] / 2
        val box1Y2 = box1[1] + box1[3] / 2

        val box2X1 = box2[0] - box2[2] / 2
        val box2Y1 = box2[1] - box2[3] / 2
        val box2X2 = box2[0] + box2[2] / 2
        val box2Y2 = box2[1] + box2[3] / 2

        // Calculate area of each box
        val box1Area = box1[2] * box1[3]
        val box2Area = box2[2] * box2[3]

        // Calculate intersection coordinates
        val xMin = maxOf(box1X1, box2X1)
        val yMin = maxOf(box1Y1, box2Y1)
        val xMax = minOf(box1X2, box2X2)
        val yMax = minOf(box1Y2, box2Y2)

        // If boxes don't intersect
        if (xMax < xMin || yMax < yMin) return 0f

        // Calculate intersection area
        val intersectionArea = (xMax - xMin) * (yMax - yMin)

        // Calculate IoU
        val unionArea = box1Area + box2Area - intersectionArea
        return intersectionArea / unionArea
    }

    fun close() {
        tfliteG1?.close()
        tfliteG1 = null
        tfliteG2?.close()
        tfliteG2 = null
    }
}