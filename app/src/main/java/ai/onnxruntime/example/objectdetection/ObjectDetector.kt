package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.InputStream
import java.nio.ByteBuffer
import java.util.Collections

internal data class Result(
    var outputBitmap: Bitmap,
    var outputBox: Array<FloatArray>
) {}

// ObjectDetector class converts the image to a byte array,
// creates an ONNX tensor, and runs the model to get the detection results

internal class ObjectDetector(
) {
    // OrtEnvironment and OrtSession are used to create an inference session for the model
    fun detect(inputStream: InputStream, ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
        // Step 1: convert image into byte array (raw image bytes)
        val rawImageBytes = inputStream.readBytes()

        // Step 2: get the shape of the byte array and make ort tensor
        val shape = longArrayOf(rawImageBytes.size.toLong())

        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            ByteBuffer.wrap(rawImageBytes),
            shape,
            OnnxJavaType.UINT8
        )
        inputTensor.use {
            // Step 3: call ort inferenceSession run
            val output = ortSession.run(
                Collections.singletonMap("image", inputTensor),
                setOf("image_out", "scaled_box_out_next")
            )

            // Step 4: output analysis
            output.use {
                val rawOutput = (output?.get(0)?.value) as ByteArray
                val boxOutput = (output?.get(1)?.value) as Array<FloatArray>
                val outputImageBitmap = byteArrayToBitmap(rawOutput)

                // Step 5: set output result
                var result = Result(outputImageBitmap, boxOutput)
                return result
            }
        }
    }

    private fun byteArrayToBitmap(data: ByteArray): Bitmap {
        return BitmapFactory.decodeByteArray(data, 0, data.size)
    }
}