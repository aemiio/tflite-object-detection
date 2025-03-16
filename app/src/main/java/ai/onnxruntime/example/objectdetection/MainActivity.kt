package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.*
import ai.onnxruntime.example.objectdetection.databinding.ActivityMainBinding
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import java.io.InputStream
import java.util.*
import android.view.View
import android.widget.SeekBar

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private var imageid = 0
    private lateinit var classes: List<String>
    private var confidenceThreshold = 0.25f

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Load initial image
        binding.imageView1.setImageBitmap(
            BitmapFactory.decodeStream(readInputImage())
        )
        imageid = 0
        classes = readClasses()

        // Set up the threshold slider
        binding.thresholdSlider.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                confidenceThreshold = progress / 100f
                binding.thresholdValue.text = String.format("%.2f", confidenceThreshold)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar) {}

            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })
        binding.thresholdSlider.progress = (confidenceThreshold * 100).toInt()
        binding.thresholdValue.text = String.format("%.2f", confidenceThreshold)

        // Initialize the model
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        binding.objectDetectionButton.setOnClickListener {
            runObjectDetection()
        }
    }

    private fun runObjectDetection() {
        // Show loading indicator
        binding.progressBar.visibility = View.VISIBLE

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val imagestream = readInputImage()

                // Update UI with the input image
                withContext(Dispatchers.Main) {
                    binding.imageView1.setImageBitmap(
                        BitmapFactory.decodeStream(imagestream)
                    )
                }
                imagestream.reset()

                // Run detection in background
                val objDetector = ObjectDetector()
                val result =
                    objDetector.detect(imagestream, ortEnv, ortSession, confidenceThreshold)

                // Update UI on main thread
                withContext(Dispatchers.Main) {
                    updateUI(result)
                    binding.progressBar.visibility = View.GONE
                    Toast.makeText(baseContext, "ObjectDetection performed!", Toast.LENGTH_SHORT)
                        .show()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    binding.progressBar.visibility = View.GONE
                    Log.e(TAG, "Exception caught when perform ObjectDetection", e)
                    Toast.makeText(
                        baseContext,
                        "Failed to perform ObjectDetection",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }

    private fun updateUI(result: Result) {
        val mutableBitmap: Bitmap = result.outputBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        // Add these debug lines
        val detectionResults = StringBuilder()
        detectionResults.append("Detected ${result.outputBox.size} objects\n\n")

        // If no detections, add debug info and set lower threshold
        if (result.outputBox.isEmpty()) {
            detectionResults.append("No objects detected above threshold: $confidenceThreshold\n")
            detectionResults.append("Try lowering the threshold value or check post-processing logic\n")
            binding.imageView2.setImageBitmap(mutableBitmap)
            binding.detectionResultsText.text = detectionResults.toString()
            return
        }

        // Existing drawing code
        val boxPaint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 5f
        }

        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 40f
            style = Paint.Style.FILL
        }

        val textBgPaint = Paint().apply {
            color = Color.BLACK
            alpha = 160
            style = Paint.Style.FILL
        }

        canvas.drawBitmap(mutableBitmap, 0.0f, 0.0f, null)

        // Process each detection
        for (box in result.outputBox) {
            // Convert centerX, centerY, width, height to left, top, right, bottom
            val left = box[0] - box[2] / 2
            val top = box[1] - box[3] / 2
            val right = box[0] + box[2] / 2
            val bottom = box[1] + box[3] / 2

            // Log detection details
            Log.d(TAG, "Drawing box: left=$left, top=$top, right=$right, bottom=$bottom")

            // Draw bounding box
            canvas.drawRect(left, top, right, bottom, boxPaint)

            // Prepare text label
            val classIdx = box[5].toInt()
            if (classIdx >= 0 && classIdx < classes.size) {
                val label = classes[classIdx]
                val confidence = box[4]
                val text = "$label: %.2f".format(confidence)

                // Draw text background
                val textWidth = textPaint.measureText(text)
                val textHeight = textPaint.textSize
                canvas.drawRect(left, top - textHeight - 5, left + textWidth + 10, top, textBgPaint)

                // Draw text
                canvas.drawText(text, left + 5, top - 5, textPaint)

                // Add to text results
                detectionResults.append("Class: $label, Confidence: %.2f%%\n".format(confidence * 100))
                detectionResults.append("Position: [x=${box[0].toInt()}, y=${box[1].toInt()}], Size: [w=${box[2].toInt()}, h=${box[3].toInt()}]\n\n")
            } else {
                // Add error info
                detectionResults.append("Error: Invalid class index $classIdx (max: ${classes.size-1})\n")
            }
        }

        binding.imageView2.setImageBitmap(mutableBitmap)
        binding.detectionResultsText.text = detectionResults.toString()
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.grade1_model
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readClasses(): List<String> {
        return resources.openRawResource(R.raw.classes).bufferedReader().readLines()
    }

    private fun readInputImage(): InputStream {
        // Choose which naming pattern to use
        val useTestImages = true // set to false to use braille_sample files

        val imageCount = 5 // Update this to match how many images you have
        imageid = (imageid + 1) % imageCount
        val fileNumber = imageid + 1 // Use 1-based numbering (1,2,3,4,5)

        // Try both file formats (jpg first, then png)
        return try {
            val fileName = if (useTestImages)
                "test_object_detection_${fileNumber}.jpg"
            else
                "braille_sample_${fileNumber}.jpg"

            assets.open(fileName)
        } catch (e: java.io.FileNotFoundException) {
            try {
                val fileName = if (useTestImages)
                    "test_object_detection_${fileNumber}.png"
                else
                    "test_object_detection_${fileNumber}.png"

                assets.open(fileName)
            } catch (e: java.io.FileNotFoundException) {
                // If no image files found, use a sample from resources
                assets.open("test_object_detection_1.jpg")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

    companion object {
        const val TAG = "ORTObjectDetection"
    }
}