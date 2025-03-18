package ai.onnxruntime.example.objectdetection

import android.content.Context

/**
 * Helper class for handling class name retrieval and related functions
 */
object BrailleClass {
    private const val TAG = "BrailleClass"

    // Get class name from class ID
    fun getClassName(classId: Int, context: Context, currentModel: String, classes: List<String>): String {
        return if (currentModel == ObjectDetector.BOTH_MODELS) {
            val bothModelsMerger = BothModelsMerger()
            val actualClassId = bothModelsMerger.getActualClassId(classId)
            val isG2 = bothModelsMerger.isG2Detection(classId)

            if (isG2) {
                // For G2 detections
                val g1ClassCount = context.resources.openRawResource(R.raw.g1_classes)
                    .bufferedReader().readLines().size
                if (actualClassId >= 0 && actualClassId < (classes.size - g1ClassCount)) {
                    classes[g1ClassCount + actualClassId]
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