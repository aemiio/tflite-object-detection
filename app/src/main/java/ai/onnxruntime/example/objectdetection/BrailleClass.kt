package ai.onnxruntime.example.objectdetection

import android.content.Context
import java.io.File

/**
 * Helper class for handling class name retrieval and related functions
 */
object BrailleClass {
    private const val TAG = "BrailleClass"

    fun getClassName(classId: Int, currentModel: String, classes: List<String>): String {
        return if (currentModel == ObjectDetector.BOTH_MODELS) {
            val bothModelsMerger = BothModelsMerger()
            val actualClassId = bothModelsMerger.getActualClassId(classId)
            val isG2 = bothModelsMerger.isG2Detection(classId)

            if (isG2) {
                // For G2 detections
                if (actualClassId >= 0 && actualClassId < classes.size / 2) {
                    classes[classes.size / 2 + actualClassId]
                } else {
                    "Unknown G2"
                }
            } else {
                // For G1 detections
                if (actualClassId >= 0 && actualClassId < classes.size / 2) {
                    classes[actualClassId]
                } else {
                    "Unknown G1"
                }
            }
        } else {
            // Standard single-model behavior
            if (classId >= 0 && classId < classes.size) classes[classId] else "Unknown"
        }
    }
}