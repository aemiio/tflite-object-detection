package ai.onnxruntime.example.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.util.Log

data class ProcessedDetectionResult(
    val displayBitmap: Bitmap,
    val detectionText: String,
    val translatedText: String
)

object BraillePostProcessor {
    private const val TAG = "BraillePostProcessor"
    private var capitalizeNext = false
    private var numberMode = false
    private var prefixDetected = false
    private var prefixText = ""

    // Reset state variables
    fun resetStates() {
        capitalizeNext = false
        numberMode = false
        prefixDetected = false
        prefixText = ""
    }

    // Process detection results from model
    fun processDetections(
        result: Result,
        context: Context,
        currentModel: String,
        classes: List<String>,
        boxPaint: Paint,
        textPaint: Paint
    ): ProcessedDetectionResult {
        Log.d(TAG, "Processing ${result.outputBox.size} detections")

        val bitmap = result.outputBitmap
        val displayWidth = bitmap.width.toFloat()
        val displayHeight = bitmap.height.toFloat()
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        // Process all detections without filtering prefixes
        val processedDetections = mutableListOf<Map<String, Any>>()
        val classDetailsMap = mutableListOf<Map<String, String>>()

        for (i in result.outputBox.indices) {
            val box = result.outputBox[i]
            Log.d(TAG, "Raw box $i values: ${box.contentToString()}")

            val detection = BrailleResult.processRawDetection(box, displayWidth, displayHeight)
            val classId = detection["classId"] as Int
            val classDetails =
                BrailleResult.getClassDetails(classId, context, currentModel, classes)

            // Store all detections, even prefixes
            processedDetections.add(detection)
            classDetailsMap.add(classDetails)
        }

        // Use BrailleFormatter to handle the results
        val brailleFormatter = BrailleFormatter()
        val cells = brailleFormatter.convertToBrailleCells(
            processedDetections,
            classDetailsMap,
            currentModel
        )

        // Apply sorting to ensure correct reading order
        val sortedLines = brailleFormatter.organizeCellsByLines(cells)
        val sortedCells = sortedLines.flatten()

        // Draw boxes on the bitmap
        brailleFormatter.drawDetectionBoxes(canvas, sortedCells, boxPaint, textPaint)

        // Format and display results
        val detectionText = brailleFormatter.formatDetectionResults(sortedCells)
        val translatedText = brailleFormatter.formatTranslatedText(sortedCells, currentModel)

        return ProcessedDetectionResult(
            displayBitmap = mutableBitmap,
            detectionText = detectionText,
            translatedText = translatedText
        )
    }
}

