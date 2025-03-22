package ai.onnxruntime.example.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Color
import android.util.Log
import kotlin.text.toInt
import kotlin.times

data class ProcessedDetectionResult(
    val displayBitmap: Bitmap,
    val detectionText: String,
    val translatedText: String
)

object BraillePostProcessor {
    private const val TAG = "BraillePostProcessor"

    fun processDetections(
        result: Result,
        context: Context,
        currentModel: String,
        classes: List<String>,
        boxPaint: Paint,
        textPaint: Paint
    ): ProcessedDetectionResult {
        Log.d(TAG, "Processing ${result.outputBox.size} detections")

        // Get scaling info from ObjectDetector
        val scaleFactor = ObjectDetector.getScaleFactor()
        val offsetX = ObjectDetector.getOffsetX()
        val offsetY = ObjectDetector.getOffsetY()

        // Get the preprocessed input bitmap
        val preprocessedBitmap = ObjectDetector.getPreprocessedInputBitmap()
            ?: result.outputBitmap

        // Create annotated result bitmap (passing currentModel parameter)
        val fullResultBitmap = createAnnotatedResultBitmap(result, offsetX, offsetY, currentModel, context)

        // Calculate content dimensions (non-padded area)
        val contentWidth = preprocessedBitmap.width - (2 * offsetX)
        val contentHeight = preprocessedBitmap.height - (2 * offsetY)

        // Crop the result bitmap to remove padding
        val displayBitmap = Bitmap.createBitmap(
            fullResultBitmap,
            offsetX,
            offsetY,
            contentWidth,
            contentHeight
        )

        // Calculate original image dimensions
        val originalWidth = kotlin.math.max(1, (contentWidth / scaleFactor).toInt())
        val originalHeight = kotlin.math.max(1, (contentHeight / scaleFactor).toInt())

        // Process detections for text output
        val processedDetections = mutableListOf<Map<String, Any>>()
        val classDetailsMap = mutableListOf<Map<String, String>>()

        for (i in result.outputBox.indices) {
            val box = result.outputBox[i]

            val detection = BrailleResult.processRawDetection(
                box,
                originalWidth.toFloat(),
                originalHeight.toFloat()
            )

            val classId = detection["classId"] as Int
            val classDetails = BrailleResult.getClassDetails(
                classId,
                context,
                currentModel,
                classes
            )

            processedDetections.add(detection)
            classDetailsMap.add(classDetails)
        }

        // Format detection results for text display
        val brailleFormatter = BrailleFormatter()
        val cells = brailleFormatter.convertToBrailleCells(
            processedDetections,
            classDetailsMap,
            currentModel
        )

        val sortedLines = brailleFormatter.organizeCellsByLines(cells)
        val sortedCells = sortedLines.flatten()

        val detectionText = brailleFormatter.formatDetectionResults(sortedCells)
        val translatedText = brailleFormatter.formatTranslatedText(sortedCells, currentModel)

        return ProcessedDetectionResult(
            displayBitmap = displayBitmap,
            detectionText = detectionText,
            translatedText = translatedText
        )
    }

    private fun createAnnotatedResultBitmap(
        result: Result,
        offsetX: Int,
        offsetY: Int,
        currentModel: String,
        context: Context?
    ): Bitmap {
        val inputSize = ObjectDetector.INPUT_SIZE
        val preprocessedBitmap = ObjectDetector.getPreprocessedInputBitmap() ?:
        return result.outputBitmap.copy(Bitmap.Config.ARGB_8888, false)

        val resultBitmap = preprocessedBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(resultBitmap)

        // Paint for detection boxes
        val boxPaint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 3f
        }

        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 20f
            setShadowLayer(2f, 0f, 0f, Color.BLACK)
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }

        // Draw boxes with braille character labels
        for (box in result.outputBox) {
            // Use the stored model space coordinates
            val modelX = if (box.size > 6) box[6] else (box[0] * ObjectDetector.getScaleFactor() + offsetX)
            val modelY = if (box.size > 7) box[7] else (box[1] * ObjectDetector.getScaleFactor() + offsetY)
            val modelW = if (box.size > 8) box[8] else (box[2] * ObjectDetector.getScaleFactor())
            val modelH = if (box.size > 9) box[9] else (box[3] * ObjectDetector.getScaleFactor())

            val left = modelX - modelW/2
            val top = modelY - modelH/2
            val right = modelX + modelW/2
            val bottom = modelY + modelH/2

            canvas.drawRect(left, top, right, bottom, boxPaint)

            // Get meaningful braille character name
            val classId = box[5].toInt()
            val conf = box[4]

            val grade: Int
            val actualClassId: Int

            when (currentModel) {
                ObjectDetector.MODEL_G2 -> {
                    // When using G2 model exclusively, all detections are G2
                    grade = 2
                    actualClassId = classId
                }
                ObjectDetector.BOTH_MODELS -> {
                    // For combined model, use the offset to determine grade
                    val isG2 = classId >= BothModelsMerger.G2_CLASS_OFFSET
                    grade = if (isG2) 2 else 1
                    actualClassId = if (isG2) classId - BothModelsMerger.G2_CLASS_OFFSET else classId
                }
                else -> {
                    // Default to G1
                    grade = 1
                    actualClassId = classId
                }
            }

            // Get the meaning for the class ID using the correct grade
            val meaning = BrailleClassIdMapper.getMeaning(actualClassId, grade)

            val label = "$meaning: ${(conf * 100).toInt()}%"
            canvas.drawText(label, left, top - 5f, textPaint)
        }

        return resultBitmap
    }
}