package ai.onnxruntime.example.objectdetection.tests

import ai.onnxruntime.example.objectdetection.*
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Paint
import androidx.core.graphics.createBitmap

class MockContext : Context() {
    override fun getResources(): android.content.res.Resources {
        return MockResources()
    }

    override fun getAssets() = null
    override fun getPackageManager() = null
    override fun getContentResolver() = null
    override fun getMainLooper() = null
    override fun getApplicationContext() = this
    override fun getPackageName() = "test.package"

    override fun getTheme() = null
    override fun getClassLoader() = javaClass.classLoader
    override fun getSystemService(name: String) = null
    override fun checkPermission(permission: String, pid: Int, uid: Int) = 0
    override fun checkCallingPermission(permission: String) = 0
    override fun checkCallingOrSelfPermission(permission: String) = 0
    override fun checkSelfPermission(permission: String) = 0
    override fun enforcePermission(permission: String, pid: Int, uid: Int, message: String?) {}
    override fun enforceCallingPermission(permission: String, message: String?) {}
    override fun enforceCallingOrSelfPermission(permission: String, message: String?) {}
}

class MockResources() : android.content.res.Resources(
    android.content.res.AssetManager(),
    android.util.DisplayMetrics(),
    android.content.res.Configuration()
) {

    override fun getString(id: Int): String = "mock_string"
    override fun getString(id: Int, vararg formatArgs: Any): String = "mock_string_formatted"
}

fun testBraillePostProcessor() {
    println("Running BraillePostProcessor Test Suite")

    // Test setup - create paints and bitmap
    val boxPaint = Paint().apply { color = 0xFF0000FF.toInt() }
    val textPaint = Paint().apply { color = 0xFFFFFFFF.toInt() }
    val bitmap = createBitmap(640, 480)
    val context = MockContext()

    // Test cases
    testBasicG1Detection(bitmap, context, boxPaint, textPaint)
    testBasicG2Detection(bitmap, context, boxPaint, textPaint)
    testBothModelsDetection(bitmap, context, boxPaint, textPaint)
    testPrefixHandling(bitmap, context, boxPaint, textPaint)
    testMultiCellBinaryPattern(bitmap, context, boxPaint, textPaint)
    testWholeWordVsPartWord(bitmap, context, boxPaint, textPaint)

    println("All tests completed")
}

fun testBasicG1Detection(bitmap: Bitmap, context: Context, boxPaint: Paint, textPaint: Paint) {
    println("Test 1: Basic Grade 1 Detection")

    // Mocked result with Grade 1 characters: h e l l o
    // x1, y1, x2, y2, score, class - float array format
    val result = Result(
        outputBox = listOf(
            floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 8.0f),  // h
            floatArrayOf(0.3f, 0.1f, 0.4f, 0.2f, 0.90f, 5.0f),  // e
            floatArrayOf(0.5f, 0.1f, 0.6f, 0.2f, 0.93f, 12.0f), // l
            floatArrayOf(0.7f, 0.1f, 0.8f, 0.2f, 0.92f, 12.0f), // l
            floatArrayOf(0.9f, 0.1f, 1.0f, 0.2f, 0.91f, 15.0f)  // o
        ),
        outputBitmap = bitmap
    )

    // Define expected classes for G1
    val g1Classes = (1..26).map { it.toString() }

    val resultObj = BraillePostProcessor.processDetections(
        result = result,
        context = context,
        currentModel = ObjectDetector.MODEL_G1,
        classes = g1Classes,
        boxPaint = boxPaint,
        textPaint = textPaint
    )

    // Verify detection text contains correct binary patterns
    assert(resultObj.detectionText.contains("h") && resultObj.detectionText.contains("e") &&
            resultObj.detectionText.contains("l") && resultObj.detectionText.contains("o"))

    // Verify translated text
    assert(resultObj.translatedText.contains("hello")) { "Expected 'hello' in translated text" }

    println("Basic Grade 1 Detection Test: PASSED")
}

fun testBasicG2Detection(bitmap: Bitmap, context: Context, boxPaint: Paint, textPaint: Paint) {
    println("Test 2: Basic Grade 2 Detection")

    // Mocked result with Grade 2 contractions
    val result = Result(
        outputBox = listOf(
            floatArrayOf(0.1f, 0.3f, 0.2f, 0.4f, 0.97f, 74.0f),  // sang-ayon

            floatArrayOf(0.3f, 0.3f, 0.4f, 0.4f, 0.93f, 19.0f),  // dot_5
            floatArrayOf(0.5f, 0.3f, 0.6f, 0.4f, 0.91f, 29.0f)   // hapon
        ),
        outputBitmap = bitmap
    )

    // Define expected classes for G2
    val g2Classes = (1..80).map { it.toString() }

    val resultObj = BraillePostProcessor.processDetections(
        result = result,
        context = context,
        currentModel = ObjectDetector.MODEL_G2,
        classes = g2Classes,
        boxPaint = boxPaint,
        textPaint = textPaint
    )

    // Verify detection includes all contractions
    assert(resultObj.detectionText.contains("sang-ayon") &&
            resultObj.detectionText.contains("dot_5") &&
            resultObj.detectionText.contains("hapon")
    ) { "Expected all contractions in detection text" }

    // Verify translated text includes proper words with formatting
    assert(resultObj.translatedText.contains("sang-ayon hapon")) {
        "Expected properly formatted translation"
    }

    println("Basic Grade 2 Detection Test: PASSED")
}

fun testBothModelsDetection(bitmap: Bitmap, context: Context, boxPaint: Paint, textPaint: Paint) {
    println("Test 3: Both Models Detection")

    // Test with a mix of G1 and G2 characters
    // Using BothModelsMerger.G2_CLASS_OFFSET (1000) to indicate G2 classes
    val result = Result(
        outputBox = listOf(
            floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 48.0f),    // i (G1)
            floatArrayOf(0.3f, 0.1f, 0.4f, 0.2f, 0.90f, 1031.0f)  // ibig (G2)
        ),
        outputBitmap = bitmap
    )

    // Define classes for both models
    val bothClasses = (1..1080).map { it.toString() }

    val resultObj = BraillePostProcessor.processDetections(
        result = result,
        context = context,
        currentModel = ObjectDetector.BOTH_MODELS,
        classes = bothClasses,
        boxPaint = boxPaint,
        textPaint = textPaint
    )

    // Verify detection shows both G1 and G2 characters
    assert(resultObj.detectionText.contains("i") && resultObj.detectionText.contains("ibig"))
    { "Expected 'i' and 'ibig' in detection text in both models mode" }

    // Verify translated text has proper spacing
    assert(resultObj.translatedText.contains("ibig")) { "Expected 'iibig' in translated text" }

    println("Both Models Detection Test: PASSED")
}

fun testPrefixHandling(bitmap: Bitmap, context: Context, boxPaint: Paint, textPaint: Paint) {
    println("Test 4: Prefix Handling")

    testG1PrefixHandling(bitmap, context, boxPaint, textPaint)
    testG2PrefixHandling(bitmap, context, boxPaint, textPaint)
    testBothModelsPrefixHandling(bitmap, context, boxPaint, textPaint)
}

fun testG1PrefixHandling(bitmap: Bitmap, context: Context, boxPaint: Paint, textPaint: Paint) {
    println("Test 4a: G1 Prefix Handling")

    // Test capital and number prefixes in G1
    val result = Result(
        outputBox = listOf(
            // Test 1: capital + a -> A
            floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 27.0f),  // capital/dot_6
            floatArrayOf(0.2f, 0.1f, 0.3f, 0.2f, 0.90f, 1.0f),   // a -> A

            // Test 2: number + a -> 1
            floatArrayOf(0.5f, 0.1f, 0.6f, 0.2f, 0.93f, 35.0f),  // number sign
            floatArrayOf(0.6f, 0.1f, 0.7f, 0.2f, 0.92f, 1.0f)    // a -> 1
        ),
        outputBitmap = bitmap
    )

    val g1Classes = (1..40).map { it.toString() }

    val resultObj = BraillePostProcessor.processDetections(
        result = result,
        context = context,
        currentModel = ObjectDetector.MODEL_G1,
        classes = g1Classes,
        boxPaint = boxPaint,
        textPaint = textPaint
    )

    // Verify capital and number prefixes work in G1
    assert(resultObj.translatedText.contains("A") && resultObj.translatedText.contains("1")) {
        "Expected capitalized 'A' and number '1' in G1 translated text"
    }

    println("G1 Prefix Handling Test: PASSED")
}

fun testG2PrefixHandling(bitmap: Bitmap, context: Context, boxPaint: Paint, textPaint: Paint) {
    println("Test 4b: G2 Prefix Handling")

    // Test dot_4 and dot_5 prefixes in G2
    val result = Result(
        outputBox = listOf(
            // Test 1: dot_4 + n -> ñ
            floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 42.0f),  // dot_4
            floatArrayOf(0.2f, 0.1f, 0.3f, 0.2f, 0.90f, 14.0f),  // n -> ñ

            // Test 2: dot_4 + another letter (not n) should remain separate
            floatArrayOf(0.4f, 0.1f, 0.5f, 0.2f, 0.93f, 42.0f),  // dot_4
            floatArrayOf(0.5f, 0.1f, 0.6f, 0.2f, 0.92f, 1.0f),   // a (should remain separate)

            // Test 3: dot_5 + noon
            floatArrayOf(0.7f, 0.1f, 0.8f, 0.2f, 0.91f, 19.0f),  // dot_5
            floatArrayOf(0.8f, 0.1f, 0.9f, 0.2f, 0.94f, 60.0f),  // noon

            // Test 4: dot_5 + opo
            floatArrayOf(0.1f, 0.3f, 0.2f, 0.4f, 0.89f, 19.0f),  // dot_5
            floatArrayOf(0.2f, 0.3f, 0.3f, 0.4f, 0.93f, 61.0f)   // opo
        ),
        outputBitmap = bitmap
    )

    val g2Classes = (1..65).map { it.toString() }

    val resultObj = BraillePostProcessor.processDetections(
        result = result,
        context = context,
        currentModel = ObjectDetector.MODEL_G2,
        classes = g2Classes,
        boxPaint = boxPaint,
        textPaint = textPaint
    )

    // Verify dot_4 and dot_5 prefixes work in G2
    assert(resultObj.translatedText.contains("ñ") &&
            resultObj.detectionText.contains("dot_4") &&
            resultObj.translatedText.contains("noon") &&
            resultObj.translatedText.contains("opo"))
    { "Expected 'ñ', and 'noon'/'opo' with proper prefixes in G2 translated text" }

    println("G2 Prefix Handling Test: PASSED")
}

fun testBothModelsPrefixHandling(bitmap: Bitmap, context: Context, boxPaint: Paint, textPaint: Paint) {
    println("Test 4c: Both Models Prefix Handling")

    // Test all prefix types in BOTH_MODELS mode
    // Using offset 1000 for G2 classes
    val result = Result(
        outputBox = listOf(
            // G1: capital + a -> A
            floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 27.0f),    // capital/dot_6 (G1)
            floatArrayOf(0.2f, 0.1f, 0.3f, 0.2f, 0.90f, 1.0f),     // a -> A (G1)

            // G2: dot_4 + n -> ñ
            floatArrayOf(0.4f, 0.1f, 0.5f, 0.2f, 0.93f, 1042.0f),  // dot_4 (G2)
            floatArrayOf(0.5f, 0.1f, 0.6f, 0.2f, 0.92f, 1014.0f),  // n -> ñ (G2)

            // G2: dot_5 + noon
            floatArrayOf(0.7f, 0.1f, 0.8f, 0.2f, 0.91f, 1019.0f),  // dot_5 (G2)
            floatArrayOf(0.8f, 0.1f, 0.9f, 0.2f, 0.94f, 1060.0f),  // noon (G2)

            // G1: number + b -> 2
            floatArrayOf(0.1f, 0.3f, 0.2f, 0.4f, 0.89f, 35.0f),    // number sign (G1)
            floatArrayOf(0.2f, 0.3f, 0.3f, 0.4f, 0.93f, 2.0f)      // b -> 2 (G1)
        ),
        outputBitmap = bitmap
    )

    val bothClasses = (1..1080).map { it.toString() }

    val resultObj = BraillePostProcessor.processDetections(
        result = result,
        context = context,
        currentModel = ObjectDetector.BOTH_MODELS,
        classes = bothClasses,
        boxPaint = boxPaint,
        textPaint = textPaint
    )

    // Verify all prefix types work in BOTH_MODELS mode
    assert(resultObj.translatedText.contains("A") &&
            resultObj.translatedText.contains("ñ") &&
            resultObj.translatedText.contains("noon") &&
            resultObj.translatedText.contains("2"))
    { "Expected 'A', 'ñ', 'noon', and '2' in BOTH_MODELS translated text" }

    println("Both Models Prefix Handling Test: PASSED")
}

fun testMultiCellBinaryPattern(bitmap: Bitmap, context: Context, boxPaint: Paint, textPaint: Paint) {
    println("Test 5: Multi-Cell Binary Pattern Display")

    // Test with a cell that has a multi-part binary pattern
    val result = Result(
        outputBox = listOf(
            floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 77.0f)  // talaga (000010-011110)
        ),
        outputBitmap = bitmap
    )

    val classes = (1..80).map { it.toString() }

    val resultObj = BraillePostProcessor.processDetections(
        result = result,
        context = context,
        currentModel = ObjectDetector.MODEL_G2,
        classes = classes,
        boxPaint = boxPaint,
        textPaint = textPaint
    )

    // Verify detection shows full binary pattern
    assert(resultObj.detectionText.contains("000010-011110") ||
            resultObj.detectionText.contains("talaga"))
    { "Expected multi-cell binary pattern or 'talaga' in detection text" }

    println("Multi-Cell Binary Pattern Test: PASSED")
}

fun testWholeWordVsPartWord(bitmap: Bitmap, context: Context, boxPaint: Paint, textPaint: Paint) {
    println("Test 6: Whole Word vs Part Word Handling")

    // Test with characters that should form "katha" (k + at + h + a)
    val result = Result(
        outputBox = listOf(
            floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 11.0f),    // k (G1)
            floatArrayOf(0.3f, 0.1f, 0.4f, 0.2f, 0.90f, 1040.0f),  // at (G2 part word)
            floatArrayOf(0.5f, 0.1f, 0.6f, 0.2f, 0.93f, 8.0f),     // h (G1)
            floatArrayOf(0.7f, 0.1f, 0.8f, 0.2f, 0.92f, 1.0f)      // a (G1)
        ),
        outputBitmap = bitmap
    )

    val bothClasses = (1..1080).map { it.toString() }

    val resultObj = BraillePostProcessor.processDetections(
        result = result,
        context = context,
        currentModel = ObjectDetector.BOTH_MODELS,
        classes = bothClasses,
        boxPaint = boxPaint,
        textPaint = textPaint
    )

    // Verify translated text combines part words correctly
    assert(resultObj.translatedText.contains("katha")) {
        "Expected 'katha' as a single word in translated text, got: ${resultObj.translatedText}"
    }

    println("Whole Word vs Part Word Handling Test: PASSED")
}

fun main() {
    testBraillePostProcessor()
}