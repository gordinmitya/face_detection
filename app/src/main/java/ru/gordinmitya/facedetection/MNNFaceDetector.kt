package ru.gordinmitya.facedetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import com.taobao.android.mnn.MNNForwardType
import com.taobao.android.mnn.MNNImageProcess
import com.taobao.android.mnn.MNNNetInstance

class MNNFaceDetector(
    val context: Context,
    val fileName: String,
    val size: Int = 320,
    val forwardType: MNNForwardType = MNNForwardType.FORWARD_AUTO
) {

    private var net: MNNNetInstance? = null
    private lateinit var session: MNNNetInstance.Session
    private lateinit var inputTensor: MNNNetInstance.Session.Tensor
    private lateinit var inputSize: IntArray

    fun prepare() {
        val file = AssetUtil.copyFileToCache(context, fileName)
        net = MNNNetInstance.createFromFile(file.absolutePath)
        val config = MNNNetInstance.Config().also {
            it.forwardType = forwardType.type
            it.numThread = NUM_THREADS
        }
        session = net!!.createSession(config)
        inputTensor = session.getInput(null)
        inputTensor.reshape(intArrayOf(1, 3, size, size))
        session.reshape()
        inputSize = inputTensor.dimensions
        require(inputSize[2] == inputSize[3] && inputSize[3] == size)
    }

    fun predict(input: Bitmap): Array<Face> {
        val config = MNNImageProcess.Config().also {
            it.mean = floatArrayOf(104f, 117f, 123f)
            it.normal = floatArrayOf(1f, 1f, 1f)
            it.source = MNNImageProcess.Format.RGBA
            it.dest = MNNImageProcess.Format.BGR
        }
        // If you ask me "why do we scale up image instead of making it smaller?"
        // I would say – I don't know
        val matrix = Matrix().also {
            val sx = 1f * input.width / inputSize[2]
            val sy = 1f * input.height / inputSize[3]
            it.setScale(sx, sy)
        }
        MNNImageProcess.convertBitmap(input, inputTensor, config, matrix)
        session.run()

        return FaceNative.getFaces(session)
    }

    fun release() {
        net?.release()
    }

    companion object {
        const val NUM_THREADS = 4
    }
}