package ru.gordinmitya.facedetection;

import com.taobao.android.mnn.MNNNetInstance;

public class FaceNative {
    static Face[] getFaces(MNNNetInstance.Session session) {
        return getFaces(session.getNetNativePtr(), session.getSessionNativePtr());
    }

    private native static Face[] getFaces(long netPtr, long sessionPtr);
}
