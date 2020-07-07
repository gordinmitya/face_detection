package ru.gordinmitya.facedetection

import android.annotation.SuppressLint
import android.graphics.*
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.taobao.android.mnn.MNNForwardType
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        doit()
    }

    private fun measure(count: Int, size: Int, device: MNNForwardType): Pair<Bitmap, LongArray> {
        val detector = MNNFaceDetector(this, "face.mnn", size, device)
        detector.prepare()

        var bitmap = assets.open("celebrities.jpg").use {
            BitmapFactory.decodeStream(it)
        }
        val timing = LongArray(count)
        for (i in 0 until count) {
            val start = System.currentTimeMillis()
            detector.predict(bitmap)
            timing[i] = System.currentTimeMillis() - start
        }
        val faces = detector.predict(bitmap)
        bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(bitmap)
        faces.forEach { face ->
            drawFace(canvas, face)
        }

        detector.release()
        return bitmap to timing
    }

    @SuppressLint("SetTextI18n")
    fun doit() = Thread {

        val count = 32
        val configs = listOf(
            256 to MNNForwardType.FORWARD_AUTO,
            320 to MNNForwardType.FORWARD_AUTO,
            512 to MNNForwardType.FORWARD_AUTO,
            640 to MNNForwardType.FORWARD_AUTO,

            256 to MNNForwardType.FORWARD_OPENCL,
            320 to MNNForwardType.FORWARD_OPENCL,
            512 to MNNForwardType.FORWARD_OPENCL,
            640 to MNNForwardType.FORWARD_OPENCL
        )

        var string = "avg of $count\n"
        text_time.text = string

        for (config in configs) {
            val (size, device) = config
            val (bitmap, timing) = measure(count, size, device)
            image_face.post {
                image_face.setImageBitmap(bitmap)

                string += "${size}x${size} ${device.name.drop(8)}" +
                        " a=${String.format("%.1f", timing.average())}" +
                        " mi=${timing.min()} ma=${timing.max()}\n"
                text_time.text = string
            }
        }
    }.start()

    private fun drawFace(canvas: Canvas, face: Face) {
        val w = canvas.width
        val h = canvas.height

        val radius = 2f
        val paint = Paint()

        paint.color = Color.YELLOW
        canvas.drawCircle(face.leftEye.x * w, face.leftEye.y * h, radius, paint)
        paint.color = Color.BLUE
        canvas.drawCircle(face.rightEye.x * w, face.rightEye.y * h, radius, paint)

        paint.color = Color.RED
        canvas.drawCircle(face.nose.x * w, face.nose.y * h, radius, paint)

        paint.color = Color.MAGENTA
        canvas.drawCircle(face.rightMouth.x * w, face.rightMouth.y * h, radius, paint)
        paint.color = Color.GREEN
        canvas.drawCircle(face.leftMouth.x * w, face.leftMouth.y * h, radius, paint)
    }
}
