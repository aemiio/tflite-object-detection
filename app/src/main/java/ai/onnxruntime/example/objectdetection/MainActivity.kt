package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.example.objectdetection.databinding.ActivityMainBinding
import kotlinx.coroutines.*
import java.io.InputStream
import java.util.*

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private var imageid = 0
    private lateinit var classes: List<String>

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.imageView1.setImageBitmap(
            BitmapFactory.decodeStream(readInputImage())
        )
        imageid = 0
        classes = readClasses()

        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        binding.objectDetectionButton.setOnClickListener {
            try {
                performObjectDetection(ortSession)
                Toast.makeText(baseContext, "ObjectDetection performed!", Toast.LENGTH_SHORT)
                    .show()
            } catch (e: Exception) {
                Log.e(TAG, "Exception caught when perform ObjectDetection", e)
                Toast.makeText(baseContext, "Failed to perform ObjectDetection", Toast.LENGTH_SHORT)
                    .show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

    // updates the UI with the detection results, drawing bounding boxes and labels on the image
    private fun updateUI(result: Result) {
        val mutableBitmap: Bitmap = result.outputBitmap.copy(Bitmap.Config.ARGB_8888, true)

        val canvas = Canvas(mutableBitmap)
        val paint = Paint()
        paint.color = Color.WHITE
        paint.textSize = 28f
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_OVER)

        canvas.drawBitmap(mutableBitmap, 0.0f, 0.0f, paint)
        var boxit = result.outputBox.iterator()
        while (boxit.hasNext()) {
            var box_info = boxit.next()
            canvas.drawText(
                "%s:%.2f".format(classes[box_info[5].toInt()], box_info[4]),
                box_info[0] - box_info[2] / 2, box_info[1] - box_info[3] / 2, paint
            )
        }

        binding.imageView2.setImageBitmap(mutableBitmap)
    }

    // from the res/raw folder
    //replace model
    private fun readModel(): ByteArray {
        val modelID = R.raw.yolov8n_with_pre_post_processing
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readClasses(): List<String> {
        return resources.openRawResource(R.raw.classes).bufferedReader().readLines()
    }

    private fun readInputImage(): InputStream { // reads an image from the assets directory.
        imageid = imageid.xor(2)
        return assets.open("test_object_detection_${imageid}.jpg") // update image file names
    }

    // performObjectDetection() function uses the ObjectDetector class to perform object detection
    // on the input image
    private fun performObjectDetection(ortSession: OrtSession) {
        var objDetector = ObjectDetector()
        var imagestream = readInputImage()
        binding.imageView1.setImageBitmap( // image is displayed
            BitmapFactory.decodeStream(imagestream)
        )
        imagestream.reset()
        var result = objDetector.detect(imagestream, ortEnv, ortSession)
        updateUI(result)
    }

    companion object {
        const val TAG = "ORTObjectDetection"
    }
}