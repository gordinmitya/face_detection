#include <jni.h>
#include <string.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

/*
 *
 * Code was partially taken from
 * https://github.com/biubug6/Face-Detector-1MB-with-landmark/blob/master/Face_Detector_ncnn/FaceDetector.cpp
 */

struct Box {
    float x1, y1, x2, y2;
};
struct Point {
    float x;
    float y;
};
struct Face {
    Box box;
    float s;
    Point point[5];
};

void create_anchor(std::vector<Box> &anchors, int w, int h) {
    anchors.clear();
    std::vector<std::vector<int> > feature_map(4), min_sizes(4);
    float steps[] = {8, 16, 32, 64};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(static_cast<int>(ceil(h / steps[i])));
        feature_map[i].push_back(static_cast<int>(ceil(w / steps[i])));
    }
    std::vector<int> minsize1 = {10, 16, 24};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 48};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {64, 96};
    min_sizes[2] = minsize3;
    std::vector<int> minsize4 = {128, 192, 256};
    min_sizes[3] = minsize4;


    for (int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (int l = 0; l < min_size.size(); ++l) {
                    float s_kx = min_size[l] * 1.0f / w;
                    float s_ky = min_size[l] * 1.0f / h;
                    float cx = (j + 0.5f) * steps[k] / w;
                    float cy = (i + 0.5f) * steps[k] / h;
                    Box axil = {cx, cy, s_kx, s_ky};
                    anchors.push_back(axil);
                }
            }
        }
    }
}

float iou(Box box1, Box box2) {
    float w = fmax(0.0f, fmin(box1.x2, box2.x2) - fmax(box1.x1, box2.x1));
    float h = fmax(0.0f, fmin(box1.y2, box2.y2) - fmax(box1.y1, box2.y1));

    float i = w * h;
    float u = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
              + (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
              - i;

    if (u <= 0.0) return 0.0f;
    else return i / u;
}

void nms(std::vector<Face> &faces, float NMS_THRESH) {
    std::vector<float> vArea(faces.size());
    for (size_t i = 0; i < faces.size(); ++i) {
        vArea[i] = (faces.at(i).box.x2 - faces.at(i).box.x1 + 1)
                   * (faces.at(i).box.y2 - faces.at(i).box.y1 + 1);
    }
    for (size_t i = 0; i < faces.size(); ++i) {
        for (size_t j = i + 1; j < faces.size();) {
            if (iou(faces[i].box, faces[j].box) >= NMS_THRESH) {
                faces.erase(faces.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}

inline bool cmp(Face a, Face b) {
    return a.s > b.s;
}

std::vector<Face> getFaces(MNN::Interpreter *net, MNN::Session *session) {
    float score_threshold = 0.6f;
    float nms_threshold = 0.65f;

    // get input size
    auto inputTensor = net->getSessionInput(session, nullptr);
    int input_height = inputTensor->shape()[2];
    int input_width = inputTensor->shape()[3];

    // get output data
    std::string output_tensor_loc = "output0";
    std::string output_tensor_class = "530";
    std::string output_tensor_landmark = "529";

    auto tensor_loc = net->getSessionOutput(session, output_tensor_loc.c_str());
    auto tensor_class = net->getSessionOutput(session, output_tensor_class.c_str());
    auto tensor_landmarks = net->getSessionOutput(session, output_tensor_landmark.c_str());

    MNN::Tensor tensor_loc_host(tensor_loc, tensor_loc->getDimensionType());
    MNN::Tensor tensor_class_host(tensor_class, tensor_class->getDimensionType());
    MNN::Tensor tensor_landmark_host(tensor_landmarks, tensor_landmarks->getDimensionType());

    tensor_loc->copyToHostTensor(&tensor_loc_host);
    tensor_class->copyToHostTensor(&tensor_class_host);
    tensor_landmarks->copyToHostTensor(&tensor_landmark_host);

    // post processing steps
    auto locPtr = tensor_loc_host.host<float>();
    auto classPtr = tensor_class_host.host<float>();
    auto landmarkPtr = tensor_landmark_host.host<float>();

    std::vector<Box> anchors;
    create_anchor(anchors, input_width, input_height);

    float max0 = *classPtr;
    float max1 = *(classPtr + 1);
    for (int i = 0; i < anchors.size(); i += 2) {
        if (classPtr[i] > max0)
            max0 = classPtr[i];
        if (classPtr[i + 1] > max1)
            max1 = classPtr[i + 1];
    }

    std::vector<Face> total_box;

    for (int i = 0; i < anchors.size(); ++i) {
        if (*(classPtr + 1) > score_threshold) {
            Box anchor = anchors[i];

            Box tmp;
            tmp.x1 = anchor.x1 + *locPtr * 0.1f * anchor.x2;
            tmp.y1 = anchor.y1 + *(locPtr + 1) * 0.1f * anchor.y2;
            tmp.x2 = anchor.x2 * exp(*(locPtr + 2) * 0.2f);
            tmp.y2 = anchor.y2 * exp(*(locPtr + 3) * 0.2f);

            Box f_box;
            f_box.x1 = (tmp.x1 - tmp.x2 / 2);
            if (f_box.x1 < 0)
                f_box.x1 = 0;
            f_box.y1 = (tmp.y1 - tmp.y2 / 2);
            if (f_box.y1 < 0)
                f_box.y1 = 0;
            f_box.x2 = (tmp.x1 + tmp.x2 / 2);
            if (f_box.x2 > 1)
                f_box.x2 = 1;
            f_box.y2 = (tmp.y1 + tmp.y2 / 2);
            if (f_box.y2 > 1)
                f_box.y2 = 1;

            Face result;
            result.box = f_box;
            result.s = *(classPtr + 1);

            // landmark
            for (int j = 0; j < 5; ++j) {
                result.point[j].x = anchor.x1 + *(landmarkPtr + (j << 1)) * 0.1f * anchor.x2;
                result.point[j].y = anchor.y1 + *(landmarkPtr + (j << 1) + 1) * 0.1f * anchor.y2;
            }

            total_box.push_back(result);
        }
        locPtr += 4;
        classPtr += 2;
        landmarkPtr += 10;
    }

    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, nms_threshold);

    return total_box;
}

jobject jniPoint(JNIEnv *env, Point point) {
    jclass clazz = env->FindClass("ru/gordinmitya/facedetection/Face$Point");
    jmethodID constructor = env->GetMethodID(clazz, "<init>", "(FF)V");
    return env->NewObject(clazz, constructor, point.x, point.y);
}

jobject jniBox(JNIEnv *env, Box box) {
    jclass clazz = env->FindClass("ru/gordinmitya/facedetection/Face$Rect");
    jmethodID constructor = env->GetMethodID(clazz, "<init>", "(FFFF)V");
    return env->NewObject(clazz, constructor, box.x1, box.y1, box.x2, box.y2);
}

jobject jniFace(JNIEnv *env, Face face) {
    jclass pointClass = env->FindClass("ru/gordinmitya/facedetection/Face$Point");

    jobject bBox = jniBox(env, face.box);
    jobjectArray landmarks = env->NewObjectArray(5, pointClass, nullptr);
    for (int i = 0; i < 5; ++i) {
        jobject point = jniPoint(env, face.point[i]);
        env->SetObjectArrayElement(landmarks, i, point);
    }

    jclass clazz = env->FindClass("ru/gordinmitya/facedetection/Face");
    jmethodID constructor = env->GetMethodID(clazz, "<init>",
                                             "(Lru/gordinmitya/facedetection/Face$Rect;F[Lru/gordinmitya/facedetection/Face$Point;)V");

    return env->NewObject(
            clazz,
            constructor,
            bBox,
            face.s,
            landmarks
    );
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_ru_gordinmitya_facedetection_FaceNative_getFaces(JNIEnv *env,
                                                      jclass type,
                                                      jlong netPtr,
                                                      jlong sessionPtr) {
    auto net = (MNN::Interpreter *) netPtr;
    auto session = (MNN::Session *) sessionPtr;

    std::vector<Face> faces = getFaces(net, session);

    jclass clazz = env->FindClass("ru/gordinmitya/facedetection/Face");
    jobjectArray array = env->NewObjectArray(static_cast<jsize>(faces.size()), clazz, nullptr);
    for (int i = 0; i < faces.size(); ++i) {
        jobject face = jniFace(env, faces[i]);
        env->SetObjectArrayElement(array, i, face);
    }

    return array;
}