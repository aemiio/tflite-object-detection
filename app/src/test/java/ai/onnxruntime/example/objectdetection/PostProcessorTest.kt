package ai.onnxruntime.example.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Paint
import androidx.core.graphics.createBitmap
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.junit.Assert.*
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject

@RunWith(RobolectricTestRunner::class)
class BraillePostProcessorTest {

    private lateinit var context: Context
    private lateinit var resources: android.content.res.Resources
    private lateinit var boxPaint: Paint
    private lateinit var textPaint: Paint
    private lateinit var bitmap: Bitmap

    @Before
    fun setUp() {
        context = mockk(relaxed = true)
        resources = mockk(relaxed = true)

        every { context.resources } returns resources
        every { resources.getString(any()) } returns "mock_string"


        // Initialize test objects
        boxPaint = Paint().apply { color = 0xFF0000FF.toInt() }
        textPaint = Paint().apply { color = 0xFFFFFFFF.toInt() }
        bitmap = createBitmap(640, 480)
    }

    // PASSED
    @Test
    fun testBasicG1Detection() {
        println("Test 1: Basic Grade 1 Detection")

        // Mocked result with Grade 1 characters: h e l l o
        val result = Result(
            outputBox = listOf(
                floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 47.0f),  // h
                floatArrayOf(0.3f, 0.1f, 0.4f, 0.2f, 0.90f, 43.0f),  // e
                floatArrayOf(0.5f, 0.1f, 0.6f, 0.2f, 0.93f, 51.0f), // l
                floatArrayOf(0.7f, 0.1f, 0.8f, 0.2f, 0.92f, 51.0f), // l
                floatArrayOf(0.9f, 0.1f, 1.0f, 0.2f, 0.91f, 55.0f)  // o
            ).toTypedArray(),
            outputBitmap = bitmap
        )

        val g1Classes = (1..26).map { it.toString() }

        val resultObj = BraillePostProcessor.processDetections(
            result = result,
            context = context,
            currentModel = ObjectDetector.MODEL_G1,
            classes = g1Classes,
            boxPaint = boxPaint,
            textPaint = textPaint
        )

        assertTrue("Expected 'h' in detection text", resultObj.detectionText.contains("h"))
        assertTrue("Expected 'hello' in translated text", resultObj.translatedText.contains("hello"))

        println("Detection text: ${resultObj.detectionText}")
        println("Translated text: ${resultObj.translatedText}")

        println("Basic Grade 1 Detection Test: PASSED")
    }

    @Test
    fun testBasicG2Detection() {
        println("Test 2: Basic Grade 2 Detection")

        // Mocked result with Grade 2 contractions
        val result = Result(
            outputBox = listOf(
                floatArrayOf(0.1f, 0.3f, 0.2f, 0.4f, 0.97f, 74.0f),  // sang-ayon
                floatArrayOf(0.3f, 0.3f, 0.4f, 0.4f, 0.93f, 19.0f),  // dot_5
                floatArrayOf(0.5f, 0.3f, 0.6f, 0.4f, 0.91f, 29.0f)   // hapon
            ).toTypedArray(),
            outputBitmap = bitmap
        )

        val g2Classes = (1..80).map { it.toString() }

        val resultObj = BraillePostProcessor.processDetections(
            result = result,
            context = context,
            currentModel = ObjectDetector.MODEL_G2,
            classes = g2Classes,
            boxPaint = boxPaint,
            textPaint = textPaint
        )

        assertTrue("Expected contractions in detection text",
            resultObj.detectionText.contains("sang-ayon") &&
                    resultObj.detectionText.contains("dot_5") &&
                    resultObj.detectionText.contains("hapon"))

        assertTrue("Expected properly formatted translation",
            resultObj.translatedText.contains("sang-ayon hapon"))

        println("Detection text: ${resultObj.detectionText}")
        println("Translated text: ${resultObj.translatedText}")

        println("Basic Grade 2 Detection Test: PASSED")
    }

    @Test
    fun testBothModelsDetection() {
        println("Test 3: Both Models Detection")

        // Using BothModelsMerger.G2_CLASS_OFFSET (1000) to indicate G2 classes
        val result = Result(
            outputBox = listOf(
                floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 48.0f),    // i (G1)
                floatArrayOf(0.3f, 0.1f, 0.4f, 0.2f, 0.90f, 1031.0f)  // ibig (G2)
            ).toTypedArray(),
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

        assertTrue("Expected 'i' and 'ibig' in detection text",
            resultObj.detectionText.contains("i") && resultObj.detectionText.contains("ibig"))

        assertTrue("Expected 'ibig' in translated text",
            resultObj.translatedText.contains("ibig"))

        println("Detection text: ${resultObj.detectionText}")
        println("Translated text: ${resultObj.translatedText}")

        println("Both Models Detection Test: PASSED")
    }

    // PASSED
    @Test
    fun testG1PrefixHandling() {
        println("Test 4a: G1 Prefix Handling")

        // Test capital and number prefixes in G1
        val result = Result(
            outputBox = listOf(
                // Test 1: capital + a -> A
                floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 40.0f),  // capital/dot_6
                floatArrayOf(0.2f, 0.1f, 0.3f, 0.2f, 0.90f, 37.0f),   // a -> A

                // Test 2: number + a -> 1
                floatArrayOf(0.5f, 0.1f, 0.6f, 0.2f, 0.93f, 54.0f),  // number sign
                floatArrayOf(0.6f, 0.1f, 0.7f, 0.2f, 0.92f, 37.0f)    // a -> 1
            ).toTypedArray(),
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

        println("Detection text: ${resultObj.detectionText}")
        println("Translated text: ${resultObj.translatedText}")

        assertTrue("Expected capitalized 'A' and number '1' in G1 translated text",
            resultObj.translatedText.contains("A") && resultObj.translatedText.contains("1"))


        println("G1 Prefix Handling Test: PASSED")
    }

    // PASSED
    @Test
    fun testG2PrefixHandling() {
        println("Test 4b: G2 Prefix Handling")

        // Test dot_5 prefixes in G2
        val result = Result(
            outputBox = listOf(
                // Test 1: dot_5 + ako
                floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 42.0f),  // dot_5
                floatArrayOf(0.2f, 0.1f, 0.3f, 0.2f, 0.90f, 0.0f),  // ako

                // Test 2: dot_5 + alam
                floatArrayOf(0.4f, 0.1f, 0.5f, 0.2f, 0.93f, 42.0f),  // dot_5
                floatArrayOf(0.5f, 0.1f, 0.6f, 0.2f, 0.92f, 1.0f),   // alam

                // Test 3: dot_5 + noon
                floatArrayOf(0.7f, 0.1f, 0.8f, 0.2f, 0.91f, 19.0f),  // dot_5
                floatArrayOf(0.8f, 0.1f, 0.9f, 0.2f, 0.94f, 60.0f),  // noon

                // Test 4: dot_5 + opo
                floatArrayOf(0.1f, 0.3f, 0.2f, 0.4f, 0.89f, 19.0f),  // dot_5
                floatArrayOf(0.2f, 0.3f, 0.3f, 0.4f, 0.93f, 61.0f)   // opo
            ).toTypedArray(),
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

        println("Detection text: ${resultObj.detectionText}")
        println("Translated text: ${resultObj.translatedText}")

        assertTrue("Expected proper prefixes in G2 translated text",
            resultObj.translatedText.contains("ako") &&
                    resultObj.detectionText.contains("alam") &&
                    resultObj.detectionText.contains("dot_5") &&
                    resultObj.translatedText.contains("noon") &&
                    resultObj.translatedText.contains("opo"))


        println("G2 Prefix Handling Test: PASSED")
    }

    // PASSED
    @Test
    fun testBothModelsPrefixHandling() {
        println("Test 4c: Both Models Prefix Handling")

        // Test all prefix types in BOTH_MODELS mode
        // Using offset 1000 for G2 classes
        val result = Result(
            outputBox = listOf(
                // G1: capital + a -> A
                floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 40.0f),    // capital/dot_6 (G1)
                floatArrayOf(0.2f, 0.1f, 0.3f, 0.2f, 0.90f, 37.0f),     // a -> A (G1)

                // G1: dot_4 + n -> ñ
                floatArrayOf(0.4f, 0.1f, 0.5f, 0.2f, 0.93f, 42.0f),  // dot_4 (G1)
                floatArrayOf(0.5f, 0.1f, 0.6f, 0.2f, 0.92f, 53.0f),  // n -> ñ (G1)

                // G2: dot_5 + noon
                floatArrayOf(0.7f, 0.1f, 0.8f, 0.2f, 0.91f, 1019.0f),  // dot_5 (G2)
                floatArrayOf(0.8f, 0.1f, 0.9f, 0.2f, 0.94f, 1060.0f),  // noon (G2)

                // G1: number + b -> 2
                floatArrayOf(0.1f, 0.3f, 0.2f, 0.4f, 0.89f, 54.0f),    // number sign (G1)
                floatArrayOf(0.2f, 0.3f, 0.3f, 0.4f, 0.93f, 38.0f)      // b -> 2 (G1)
            ).toTypedArray(),
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

        println("Detection text: ${resultObj.detectionText}")
        println("Translated text: ${resultObj.translatedText}")

        assertTrue("Expected 'A', 'ñ', 'noon', and '2' in BOTH_MODELS translated text",
            resultObj.translatedText.contains("A") &&
                    resultObj.translatedText.contains("ñ") &&
                    resultObj.translatedText.contains("noon") &&
                    resultObj.translatedText.contains("2"))

        println("Both Models Prefix Handling Test: PASSED")
    }

    @Test
    fun testMultiCellBinaryPattern() {
        println("Test 5: Multi-Cell Binary Pattern Display")

        // Test with a cell that has a multi-part binary pattern
        val result = Result(
            outputBox = listOf(
                floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 77.0f)  // talaga (000010-011110)
            ).toTypedArray(),
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

        assertTrue("Expected multi-cell binary pattern or 'talaga' in detection text",
            resultObj.detectionText.contains("000010-011110") ||
                    resultObj.detectionText.contains("talaga"))


        println("Detection text: ${resultObj.detectionText}")
        println("Translated text: ${resultObj.translatedText}")

        println("Multi-Cell Binary Pattern Test: PASSED")
    }


    // PASSED
    @Test
    fun testWholeWordVsPartWord() {
        println("Test 6: Whole Word vs Part Word Handling")

        // Test with characters that should form "katha" (k + at + h + a)
        val result = Result(
            outputBox = listOf(
                floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f, 0.95f, 50.0f),    // k (G1)
                floatArrayOf(0.3f, 0.1f, 0.4f, 0.2f, 0.90f, 1008.0f),  // at (G2 part word)
                floatArrayOf(0.5f, 0.1f, 0.6f, 0.2f, 0.93f, 47.0f),     // h (G1)
                floatArrayOf(0.7f, 0.1f, 0.8f, 0.2f, 0.92f, 37.0f)      // a (G1)
            ).toTypedArray(),
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

        assertTrue("Expected 'katha' as a single word in translated text",
            resultObj.translatedText.contains("katha"))


        println("Detection text: ${resultObj.detectionText}")
        println("Translated text: ${resultObj.translatedText}")

        println("Whole Word vs Part Word Handling Test: PASSED")
    }
}