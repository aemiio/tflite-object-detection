package ai.onnxruntime.example.objectdetection

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint

/**
 * Helper class for processing detection results
 */
object BrailleResult {
    // Process raw detection data
    fun processRawDetection(box: FloatArray, displayWidth: Float, displayHeight: Float): Map<String, Any> {
        val modelWidth = ObjectDetector.INPUT_SIZE.toFloat()
        val modelHeight = ObjectDetector.INPUT_SIZE.toFloat()

        val ratioWidth = displayWidth / modelWidth
        val ratioHeight = displayHeight / modelHeight

        val x = box[0] * ratioWidth
        val y = box[1] * ratioHeight
        val w = box[2] * ratioWidth
        val h = box[3] * ratioHeight
        val conf = box[4]
        val classId = box[5].toInt()

        val left = Math.max(0f, x - w / 2)
        val top = Math.max(0f, y - h / 2)
        val right = Math.min(displayWidth, x + w / 2)
        val bottom = Math.min(displayHeight, y + h / 2)

        return mapOf(
            "x" to x, "y" to y, "w" to w, "h" to h,
            "conf" to conf, "classId" to classId,
            "left" to left, "top" to top, "right" to right, "bottom" to bottom
        )
    }

    // Process class ID and get related information
    fun getClassDetails(classId: Int, context: Context, currentModel: String, classes: List<String>): Map<String, String> {
        val className = BrailleClass.getClassName(classId, context, currentModel, classes)

        // Get binary pattern and meaning based on which model was used
        val (binaryPattern, meaningText) = if (currentModel == ObjectDetector.BOTH_MODELS) {
            val bothModelsMerger = BothModelsMerger()
            val actualClassId = bothModelsMerger.getActualClassId(classId)
            val isG2 = bothModelsMerger.isG2Detection(classId)

            // Use appropriate map based on model type
            if (isG2) {
                val entry = BrailleMap.G2brailleMap[actualClassId]
                Pair(entry?.binary?.split("-")?.first() ?: "?", entry?.meaning ?: "?")
            } else {
                val entry = BrailleMap.G1brailleMap[actualClassId]
                Pair(entry?.binary?.split("-")?.first() ?: "?", entry?.meaning ?: "?")
            }
        } else if (currentModel == ObjectDetector.MODEL_G2) {
            val entry = BrailleMap.G2brailleMap[classId]
            Pair(entry?.binary?.split("-")?.first() ?: "?", entry?.meaning ?: "?")
        } else {
            val entry = BrailleMap.G1brailleMap[classId]
            Pair(entry?.binary?.split("-")?.first() ?: "?", entry?.meaning ?: "?")
        }

        return mapOf(
            "className" to className,
            "binaryPattern" to binaryPattern,
            "meaningText" to meaningText
        )
    }

    // Format results for display
    fun formatDetectionResults(detections: List<Map<String, Any>>, classDetails: List<Map<String, String>>): String {
        val detectionResults = StringBuilder()
        detectionResults.append("Found ${detections.size} detections\n\n")

        for (i in detections.indices) {
            val detection = detections[i]
            val details = classDetails[i]

            detectionResults.append("Box $i: ${details["className"]} (${(detection["conf"] as Float * 100).toInt()}%), " +
                    "Binary: ${details["binaryPattern"]}, " +
                    "Text: ${details["meaningText"]}\n")
        }

        return detectionResults.toString()
    }

    // Sort detections in proper reading order (row by row, left to right)
    fun sortDetectionsInReadingOrder(
        detections: List<Map<String, Any>>,
        classDetails: List<Map<String, String>>
    ): Pair<List<Map<String, Any>>, List<Map<String, String>>> {
        // Pair each detection with its class detail for sorting
        val combined = detections.zip(classDetails).toMutableList()

        // 1. Group boxes by rows using Y-coordinate with tolerance
        val yTolerance = 0.2f * (detections.maxOfOrNull { it["h"] as Float } ?: 30f)
        val rowGroups = mutableListOf<MutableList<Pair<Map<String, Any>, Map<String, String>>>>()

        for (item in combined) {
            val y = item.first["y"] as Float
            var assignedToRow = false

            // Try to find a row this box belongs to
            for (row in rowGroups) {
                val rowY = row.first().first["y"] as Float
                if (Math.abs(y - rowY) < yTolerance) {
                    row.add(item)
                    assignedToRow = true
                    break
                }
            }

            // If no matching row, create new row
            if (!assignedToRow) {
                rowGroups.add(mutableListOf(item))
            }
        }

        // 2. Sort each row by X-coordinate
        for (row in rowGroups) {
            row.sortBy { it.first["x"] as Float }
        }

        // 3. Sort rows by Y-coordinate
        rowGroups.sortBy { it.first().first["y"] as Float }

        // 4. Flatten back to lists
        val sortedCombined = rowGroups.flatten()
        val sortedDetections = sortedCombined.map { it.first }
        val sortedClassDetails = sortedCombined.map { it.second }

        return Pair(sortedDetections, sortedClassDetails)
    }

    // Format translated braille text
    fun formatBrailleText(classDetails: List<Map<String, String>>): String {
        val brailleTextResults = StringBuilder()
        brailleTextResults.append("Translated Braille:\n")

        for (details in classDetails) {
            brailleTextResults.append("${details["meaningText"]} ")
        }

        return brailleTextResults.toString()
    }

    // Draw detection boxes and labels on canvas
    fun drawDetectionBoxes(
        canvas: Canvas,
        detections: List<Map<String, Any>>,
        classDetails: List<Map<String, String>>,
        boxPaint: Paint,
        textPaint: Paint
    ) {
        for (i in detections.indices) {
            val detection = detections[i]
            val details = classDetails[i]

            // Draw box
            canvas.drawRect(
                detection["left"] as Float,
                detection["top"] as Float,
                detection["right"] as Float,
                detection["bottom"] as Float,
                boxPaint
            )

            // Draw label
            val label = "${details["className"]} ${(detection["conf"] as Float * 100).toInt()}%"
            canvas.drawText(
                label,
                detection["left"] as Float,
                Math.max(30f, (detection["top"] as Float) - 5f),
                textPaint
            )
        }
    }
}