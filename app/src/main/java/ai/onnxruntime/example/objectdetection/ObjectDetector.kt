package ai.onnxruntime.example.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

internal data class Result(
    var outputBitmap: Bitmap,
    var outputBox: Array<FloatArray>
)

internal class ObjectDetector {

    companion object {
        const val TAG = "ObjectDetector"
        const val INPUT_SIZE = 640
        const val NUM_THREADS = 4
    }

    private var tflite: Interpreter? = null

    fun initialize(context: Context) {
        val tfliteOptions = Interpreter.Options().apply {
            setNumThreads(NUM_THREADS)
        }

        try {
            val modelFile = context.assets.openFd("best_float32.tflite")
            val fileChannel = FileInputStream(modelFile.fileDescriptor).channel
            val modelBuffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                modelFile.startOffset,
                modelFile.declaredLength
            )
            tflite = Interpreter(modelBuffer, tfliteOptions)
            Log.d(TAG, "TFLite model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading TFLite model", e)
            throw RuntimeException("Error loading TFLite model", e)
        }
    }

    fun detect(
        inputStream: InputStream,
        context: Context,
        confidenceThreshold: Float = 0.25f
    ): Result {

        if (tflite == null) {
            initialize(context)
        }

        // Load and preprocess the image
        val originalBitmap = BitmapFactory.decodeStream(inputStream)

        // Prepare input and output buffers
        val inputBuffer = preprocessImage(originalBitmap)

        // Output shape is [1, 71, 8400] based on the model info
        val outputShape = intArrayOf(1, 71, 8400)
        val outputBuffer = Array(1) { Array(71) { FloatArray(8400) } }

        // Run inference
        tflite?.run(inputBuffer, outputBuffer)

        // Process results
        val detections = postprocessYOLOv8(
            outputBuffer[0],
            originalBitmap.width,
            originalBitmap.height,
            confidenceThreshold
        )

        return Result(originalBitmap, detections)
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        inputBuffer.rewind()

        for (pixel in pixels) {
            // TFLite model expects HWC format - already matches the model input [1,640,640,3]
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

        Log.d(TAG, "Original image dimensions: $originalWidth x $originalHeight")

        for (i in 0 until 8400) {
            var maxClassScore = 0f
            var bestClassIdx = 0

            for (c in 4 until 71) {
                val score = outputBuffer[c][i]
                if (score > maxClassScore) {
                    maxClassScore = score
                    bestClassIdx = c - 4
                }
            }

            if (maxClassScore >= confidenceThreshold) {
                // IMPORTANT: YOLOv8 outputs normalized coordinates (0-1)
                // Convert to model space (0-640)
                val x = outputBuffer[0][i] * INPUT_SIZE
                val y = outputBuffer[1][i] * INPUT_SIZE
                val w = outputBuffer[2][i] * INPUT_SIZE
                val h = outputBuffer[3][i] * INPUT_SIZE

                Log.d(TAG, "Detection in model space: x=$x, y=$y, w=$w, h=$h, conf=$maxClassScore, class=$bestClassIdx")

                rawResults.add(floatArrayOf(x, y, w, h, maxClassScore, bestClassIdx.toFloat()))
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
        tflite?.close()
        tflite = null
    }
}