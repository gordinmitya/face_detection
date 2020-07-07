# Android face detection with landmarks

Used framework – [MNN by Alibaba](https://github.com/alibaba/MNN);

Model was taken from [biubug6/Face-Detector-1MB-with-landmark (RFB)](https://github.com/biubug6/Face-Detector-1MB-with-landmark);

Then converted with [built docker image MNNConvert](https://github.com/gordinmitya/docker_that_framework/tree/master/mnn);

```bash
cd ~/Face-Detector-1MB-with-landmark
python3 convert_to_onnx.py
python3 -m onnxsim faceDetector.onnx sim.onnx

cd ~/mnn/build
./MNNConvert --bizCode MNN --MNNModel face.mnn -f ONNX --modelFile ~/Face-Detector-1MB-with-landmark/sim.onnx
cp ~/face.mnn ~/face_detection/app/src/main/assets/face.mnn
```

Face bounding box + landmarks: eyes, nose, mouth.

## Example
![example](./img/screenshot.jpg)

Timings on Snapdragon 855 (including pre and postprocessing).

## Notes

If you're going to obfuscate with Proguard don't forget to [include these rules](./app/proguard-rules.pro) to keep Face class. It's used from JNI.

There're alternatives like [TFLite with Blazeface](https://github.com/google/mediapipe).

## TODO

- [ ] show bounding box
- [ ] capture stream from camera
- [ ] GooglePlay