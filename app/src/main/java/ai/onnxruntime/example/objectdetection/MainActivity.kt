package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.example.objectdetection.databinding.ActivityMainBinding
import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.InputStream

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var imageid = 0
    private lateinit var classes: List<String>
    private var confidenceThreshold = 0.25f
    private var currentModel = ObjectDetector.MODEL_G1

    companion object {
        const val TAG = "ORTObjectDetection"
    }

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


        binding.thresholdSlider.setOnSeekBarChangeListener(object :
            SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                confidenceThreshold = progress / 100f
                binding.thresholdValue.text = String.format("%.2f", confidenceThreshold)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar) {}

            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })
        binding.thresholdSlider.progress = (confidenceThreshold * 100).toInt()
        binding.thresholdValue.text = String.format("%.2f", confidenceThreshold)

        binding.modelG1Button.setOnClickListener {
            currentModel = ObjectDetector.MODEL_G1
            binding.modelG1Button.isSelected = true
            binding.modelG2Button.isSelected = false
            classes = readClasses()
            Toast.makeText(this, "Model G1 selected", Toast.LENGTH_SHORT).show()
        }

        binding.modelG2Button.setOnClickListener {
            currentModel = ObjectDetector.MODEL_G2
            binding.modelG1Button.isSelected = false
            binding.modelG2Button.isSelected = true
            classes = readClasses()
            Toast.makeText(this, "Model G2 selected", Toast.LENGTH_SHORT).show()
        }

        binding.objectDetectionButton.setOnClickListener {
            runObjectDetection()
        }
    }

    private fun runObjectDetection() {

        binding.progressBar.visibility = View.VISIBLE

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val imagestream = readInputImage()


                withContext(Dispatchers.Main) {
                    binding.imageView1.setImageBitmap(
                        BitmapFactory.decodeStream(imagestream)
                    )
                }
                imagestream.reset()


                val objDetector = ObjectDetector()
                val result = objDetector.detect(
                    imagestream,
                    this@MainActivity,
                    confidenceThreshold,
                    currentModel
                )
                objDetector.close()


                withContext(Dispatchers.Main) {
                    updateUI(result)
                    binding.progressBar.visibility = View.GONE
                    Toast.makeText(
                        baseContext,
                        "ObjectDetection performed with ${if (currentModel == ObjectDetector.MODEL_G1) "G1" else "G2"} model!\" ",
                        Toast.LENGTH_SHORT
                    )
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
        Log.d(TAG, "Starting updateUI with ${result.outputBox.size} detections")

        // Get actual displayed image dimensions
        val bitmap = result.outputBitmap
        val displayWidth = bitmap.width.toFloat()
        val displayHeight = bitmap.height.toFloat()

        // Log dimensions for debugging
        Log.d(TAG, "Original image dimensions: ${displayWidth}x${displayHeight}")

        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        val boxPaint = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.STROKE
            strokeWidth = 5f
            isAntiAlias = true
        }

        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 25f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setShadowLayer(3f, 1f, 1f, Color.BLACK)
        }

        val detectionResults = StringBuilder()
        detectionResults.append("Found ${result.outputBox.size} detections\n\n")

        // correct approach based on how YOLOv8 outputs coordinates
        for (i in result.outputBox.indices) {
            val box = result.outputBox[i]

            // Log raw box values
            Log.d(TAG, "Raw box $i values: ${box.contentToString()}")

            // Reinterpret the coordinates based on INPUT_SIZE (640)
            // The raw x,y,w,h are relative to the 640x640 input the model processed
            val modelWidth = ObjectDetector.INPUT_SIZE.toFloat()
            val modelHeight = ObjectDetector.INPUT_SIZE.toFloat()

            // Calculate ratios to scale from model space to image space
            val ratioWidth = displayWidth / modelWidth
            val ratioHeight = displayHeight / modelHeight

            // Apply transformation
            val x = box[0] * ratioWidth
            val y = box[1] * ratioHeight
            val w = box[2] * ratioWidth
            val h = box[3] * ratioHeight
            val conf = box[4]
            val classId = box[5].toInt()

            // Calculate rectangle coordinates
            val left = Math.max(0f, x - w / 2)
            val top = Math.max(0f, y - h / 2)
            val right = Math.min(displayWidth, x + w / 2)
            val bottom = Math.min(displayHeight, y + h / 2)

            Log.d(
                TAG,
                "Drawing box $i: ($left,$top,$right,$bottom) on image ${displayWidth}x${displayHeight}"
            )

            // Draw box
            canvas.drawRect(left, top, right, bottom, boxPaint)

//            // Draw crosshair at center point for debugging
//            canvas.drawLine(x - 10, y, x + 10, y, boxPaint)
//            canvas.drawLine(x, y - 10, x, y + 10, boxPaint)

            // Add to results text
            val className =
                if (classId >= 0 && classId < classes.size) classes[classId] else "Unknown"
            detectionResults.append("Box $i: $className (${(conf * 100).toInt()}%)\n")
            detectionResults.append("  Center: (${x.toInt()}, ${y.toInt()}) Size: ${w.toInt()}x${h.toInt()}\n")

            // Draw label
            val label = "$className ${(conf * 100).toInt()}%"
            canvas.drawText(label, left, Math.max(30f, top - 5f), textPaint)
        }

        binding.imageView2.setImageBitmap(mutableBitmap)
        binding.detectionResultsText.text = detectionResults.toString()
    }

    private fun readClasses(): List<String> {
        return if (currentModel == ObjectDetector.MODEL_G2) {
            resources.openRawResource(R.raw.g2_classes).bufferedReader().readLines()
        } else {
            resources.openRawResource(R.raw.g1_classes).bufferedReader().readLines()
        }
    }

    private fun readInputImage(): InputStream {

        val useTestImages = true

        val imageCount = 5
        imageid = (imageid + 1) % imageCount
        val fileNumber = imageid + 1 // Use 1-based numbering (1,2,3,4,5)


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
                assets.open("test_object_detection_1.jpg")
            }
        }
    }
}