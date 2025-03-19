package ai.onnxruntime.example.objectdetection

import android.content.Context
import android.util.Log

object BrailleClassIdMapper {
    private const val TAG = "BrailleClassIdMapper"

    // Maps class IDs to binary patterns for different models
    private val g1ClassIdToBinaryMap = mutableMapOf<Int, String>()
    private val g2ClassIdToBinaryMap = mutableMapOf<Int, String>()

    init {
        // Initialize maps from BrailleMap data
        BrailleMap.G1brailleMap.forEach { (id, entry) ->
            // Get first part of binary pattern (for multi-cell patterns like "000001-100000")
            g1ClassIdToBinaryMap[id] = entry.binary.split("-").first()
        }

        BrailleMap.G2brailleMap.forEach { (id, entry) ->
            g2ClassIdToBinaryMap[id] = entry.binary.split("-").first()
        }
    }

    /**
     * Get binary pattern for a class ID, handling both single and combined models
     */
    fun getBinaryPattern(classId: Int): String {
        // Check if this is from the combined model
        if (classId >= BothModelsMerger.G2_CLASS_OFFSET) {
            val actualId = classId - BothModelsMerger.G2_CLASS_OFFSET
            return g2ClassIdToBinaryMap[actualId] ?: run {
                Log.w(TAG, "Unknown G2 class ID: $actualId")
                "??????"
            }
        }

        // Otherwise treat as G1 class
        return g1ClassIdToBinaryMap[classId] ?: run {
            Log.w(TAG, "Unknown G1 class ID: $classId")
            "??????"
        }
    }

    /**
     * Get meaning for a class ID
     */
    fun getMeaning(classId: Int): String {
        if (classId >= BothModelsMerger.G2_CLASS_OFFSET) {
            val actualId = classId - BothModelsMerger.G2_CLASS_OFFSET
            return BrailleMap.G2brailleMap[actualId]?.meaning ?: "?"
        }

        return BrailleMap.G1brailleMap[classId]?.meaning ?: "?"
    }

    /**
     * Debug function to help with mapping issues
     */
    fun loadMappingsFromResources(context: Context) {
        Log.d(TAG, "Initializing Braille class ID mappings...")

        // Log G1 mappings for debugging
        BrailleMap.G1brailleMap.entries.take(10).forEach { (id, entry) ->
            Log.d(TAG, "G1 Class $id: ${entry.binary} → ${entry.meaning}")
        }

        // Log G2 mappings for debugging
        BrailleMap.G2brailleMap.entries.take(10).forEach { (id, entry) ->
            Log.d(TAG, "G2 Class $id: ${entry.binary} → ${entry.meaning}")
        }

        Log.d(TAG, "Total mappings: ${g1ClassIdToBinaryMap.size} G1, ${g2ClassIdToBinaryMap.size} G2")
    }


    // Map class IDs to binary patterns and meanings directly
    fun getBrailleEntry(classId: Int, grade: Int): BrailleEntry? {
        return BrailleMap.getBrailleMap(grade)[classId]
    }

    // Get the meaning directly from the map
    fun getMeaning(classId: Int, grade: Int): String {
        return getBrailleEntry(classId, grade)?.meaning ?: "?"
    }

    // Get the binary pattern directly from the map
    fun getBinary(classId: Int, grade: Int): String {
        return getBrailleEntry(classId, grade)?.binary ?: "?"
    }
}