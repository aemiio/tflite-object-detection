package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.InputStream
import java.nio.FloatBuffer
import java.util.Collections
import java.util.Optional
import android.util.Log

internal data class Result(
    var outputBitmap: Bitmap,
    var outputBox: Array<FloatArray>
)

internal class ObjectDetector {

    companion object {
        const val TAG = "ObjectDetector"
    }

    fun detect(
        inputStream: InputStream,
        ortEnv: OrtEnvironment,
        ortSession: OrtSession,
        confidenceThreshold: Float = 0.25f
    ): Result {
        // Load and preprocess the image
        val originalBitmap = BitmapFactory.decodeStream(inputStream)
        val inputTensor = preprocessImage(originalBitmap, ortEnv)

        inputTensor.use {
            // Run inference with the correct input name "images"
            val output = ortSession.run(
                Collections.singletonMap("images", inputTensor),
                Collections.singleton("output0")
            )

            output.use {
                val outputObj = output.get("output0")
                // Handle Optional return type
                val outputTensor = if (outputObj is Optional<*>) {
                    if (outputObj.isPresent) {
                        outputObj.get() as OnnxTensor
                    } else {
                        throw RuntimeException("Output tensor is empty")
                    }
                } else {
                    outputObj as OnnxTensor
                }

                val detections = postprocessYOLOv8(
                    outputTensor,
                    originalBitmap.width,
                    originalBitmap.height,
                    confidenceThreshold
                )

                return Result(originalBitmap, detections)
            }
        }
    }

    private fun preprocessImage(bitmap: Bitmap, ortEnv: OrtEnvironment): OnnxTensor {
        val inputWidth = 640
        val inputHeight = 640

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val buffer = FloatBuffer.allocate(inputWidth * inputHeight * 3)

        val pixels = IntArray(inputWidth * inputHeight)
        resizedBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        // HWC to CHW and normalize
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val pixel = pixels[y * inputWidth + x]
                // RGB order with normalization
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f

                buffer.put(r)
                buffer.put(g)
                buffer.put(b)
            }
        }
        buffer.rewind()

        return OnnxTensor.createTensor(
            ortEnv,
            buffer,
            longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
        )
    }

    private fun postprocessYOLOv8(
        outputTensor: OnnxTensor,
        originalWidth: Int,
        originalHeight: Int,
        confidenceThreshold: Float
    ): Array<FloatArray> {
        val results = mutableListOf<FloatArray>()

        // Get raw output data
        val outputData = outputTensor.floatBuffer
        val dimensions = outputTensor.info.shape

        // Log dimensions for debugging
        Log.d(TAG, "Output tensor shape: ${dimensions.contentToString()}")

        // Debug first few values
        val debugSize = minOf(20, outputData.capacity())
        val debugValues = FloatArray(debugSize)
        val originalPosition = outputData.position()
        outputData.get(debugValues)
        Log.d(TAG, "First $debugSize values: ${debugValues.joinToString()}")
        outputData.position(originalPosition) // Reset position

        // YOLOv8 output is transposed: [1, 84, 8400] where 84 = 4(box) + 80(classes)
        // Correct indexing for this format
        val anchors = dimensions[2].toInt() // 8400
        val numValues = dimensions[1].toInt() // 84
        val numClasses = numValues - 4 // 80

        val scaleX = originalWidth / 640f
        val scaleY = originalHeight / 640f

        // Process each detection (each anchor)
        for (i in 0 until anchors) {
            // Find best class and score for this anchor
            var maxClassScore = 0f
            var bestClassIdx = 0

            // Check each class (positions 4 to end)
            for (c in 4 until numValues) {
                val score = outputData.get(c * anchors + i)
                if (score > maxClassScore) {
                    maxClassScore = score
                    bestClassIdx = c - 4 // Adjust for class index
                }
            }

            // If confidence meets threshold
            if (maxClassScore >= confidenceThreshold) {
                // Get box coordinates (in YOLOv8, these are already in xywh format)
                val x = outputData.get(0 * anchors + i) * scaleX
                val y = outputData.get(1 * anchors + i) * scaleY
                val w = outputData.get(2 * anchors + i) * scaleX
                val h = outputData.get(3 * anchors + i) * scaleY

                // Add detection to results [x, y, w, h, confidence, class]
                results.add(floatArrayOf(x, y, w, h, maxClassScore, bestClassIdx.toFloat()))
            }
        }

        Log.d(TAG, "Found ${results.size} valid detections")
        return results.toTypedArray()
    }
}