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

        BrailleClassIdMapper.loadMappingsFromResources(this)


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

        binding.bothModelsButton.setOnClickListener {
            currentModel = ObjectDetector.BOTH_MODELS
            binding.modelG1Button.isSelected = false
            binding.modelG2Button.isSelected = false
            binding.bothModelsButton.isSelected = true
            classes = readClasses()
            Toast.makeText(this, "Both Models selected", Toast.LENGTH_SHORT).show()
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

        // Process detections using BraillePostProcessor
        val processedResult = BraillePostProcessor.processDetections(
            result = result,
            context = this,
            currentModel = currentModel,
            classes = classes,
            boxPaint = boxPaint,
            textPaint = textPaint
        )

        binding.imageView2.setImageBitmap(processedResult.displayBitmap)

        // Remove debug view completely
        binding.imageViewDebug.visibility = View.GONE

        binding.detectionResultsText.text =
            processedResult.detectionText + "\n" + processedResult.translatedText
    }

    private fun readClasses(): List<String> {
        return when (currentModel) {
            ObjectDetector.MODEL_G2 -> {
                resources.openRawResource(R.raw.g2_classes).bufferedReader().readLines()
            }

            ObjectDetector.BOTH_MODELS -> {
                // Combine both class lists
                val g1Classes =
                    resources.openRawResource(R.raw.g1_classes).bufferedReader().readLines()
                val g2Classes =
                    resources.openRawResource(R.raw.g2_classes).bufferedReader().readLines()
                        .mapIndexed { index, className -> "$className" }
                g1Classes + g2Classes
            }

            else -> {
                resources.openRawResource(R.raw.g1_classes).bufferedReader().readLines()
            }
        }
    }

    // fix this if di tamarin whawha
    private fun readInputImage(): InputStream {

        val useTestImages = true

        val imageCount = 6
        imageid = (imageid + 1) % imageCount

        val fileNumber = if (imageid == 0) 0 else imageid


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