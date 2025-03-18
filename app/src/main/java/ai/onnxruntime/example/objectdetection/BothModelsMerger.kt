package ai.onnxruntime.example.objectdetection

import android.graphics.Bitmap
import android.util.Log

/**
 * Handles the merging of detection results from G1 and G2 models.
 */
class BothModelsMerger {
    companion object {
        private const val TAG = "BothModelsMerger"
        const val G2_CLASS_OFFSET = 1000 // Offset to distinguish G2 classes
    }

    /**
     * Merges results from G1 and G2 models with advanced handling of unknown detections.
     */
    fun mergeResults(
        g1Result: Result,
        g2Result: Result,
        bitmap: Bitmap
    ): Result {
        Log.d(TAG, "Merging results: ${g1Result.outputBox.size} from G1, ${g2Result.outputBox.size} from G2")

        // Create combined detection list
        val combinedDetections = mutableListOf<FloatArray>()

        // Add G1 detections
        g1Result.outputBox.forEach { detection ->
            combinedDetections.add(detection)
        }

        // Add G2 detections with class offset
        g2Result.outputBox.forEach { detection ->
            val modifiedDetection = detection.clone()
            modifiedDetection[5] = detection[5] + G2_CLASS_OFFSET
            combinedDetections.add(modifiedDetection)
        }

        // Apply NMS to remove overlapping detections
        val mergedDetections = applyNMS(combinedDetections.toTypedArray(), 0.5f)

        Log.d(TAG, "Merged to ${mergedDetections.size} detections after NMS")
        return Result(bitmap, mergedDetections.toTypedArray())
    }

    /**
     * Checks if a detection is from the G2 model based on its class ID.
     */
    fun isG2Detection(classId: Int): Boolean {
        return classId >= G2_CLASS_OFFSET
    }

    /**
     * Gets the actual class ID by removing the offset for G2 detections.
     */
    fun getActualClassId(classId: Int): Int {
        return if (isG2Detection(classId)) classId - G2_CLASS_OFFSET else classId
    }

    /**
     * Apply Non-Maximum Suppression to filter overlapping boxes.
     */
    private fun applyNMS(boxes: Array<FloatArray>, iouThreshold: Float): List<FloatArray> {
        if (boxes.isEmpty()) return emptyList()

        // Sort boxes by confidence (highest first)
        val sortedBoxes = boxes.sortedByDescending { it[4] }
        val selected = mutableListOf<FloatArray>()
        val active = BooleanArray(sortedBoxes.size) { true }

        for (i in sortedBoxes.indices) {
            if (!active[i]) continue

            selected.add(sortedBoxes[i])

            for (j in i + 1 until sortedBoxes.size) {
                if (!active[j]) continue

                if (calculateIOU(sortedBoxes[i], sortedBoxes[j]) >= iouThreshold) {
                    active[j] = false
                }
            }
        }

        return selected
    }

    /**
     * Calculate Intersection over Union between two boxes.
     */
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
}