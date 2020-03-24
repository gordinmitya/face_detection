package ru.gordinmitya.facedetection

import android.annotation.SuppressLint
import android.graphics.*
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        doit()
    }

    @SuppressLint("SetTextI18n")
    fun doit() = Thread {
        val detector = MNNFaceDetector(this, "face.mnn")
        detector.prepare()

        var bitmap = assets.open("celebrities.jpg").use {
            BitmapFactory.decodeStream(it)
        }
        val count = 32
        var sum = 0L
        for (i in 0 until count) {
            val start = System.currentTimeMillis()
            detector.predict(bitmap)
            sum += System.currentTimeMillis() - start
        }
        val avg = 1.0 * sum / count
        val faces = detector.predict(bitmap)
        bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(bitmap)
        faces.forEach { face ->
            drawFace(canvas, face)
        }
        image_face.post {
            image_face.setImageBitmap(bitmap)
            text_time.text = "avg of $count runs = ${String.format("%.1f", avg)}ms"
        }

        detector.release()
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
