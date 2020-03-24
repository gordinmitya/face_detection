package ru.gordinmitya.facedetection;

public class Face {
    public final Rect bBox;
    public final float score;
    public final Point leftEye;
    public final Point rightEye;
    public final Point nose;
    public final Point leftMouth;
    public final Point rightMouth;

    public Face(Rect bBox, float score, Point[] landmarks) {
        this.bBox = bBox;
        this.score = score;
        this.leftEye = landmarks[0];
        this.rightEye = landmarks[1];
        this.nose = landmarks[2];
        this.leftMouth = landmarks[3];
        this.rightMouth = landmarks[4];
    }

    public static class Rect {
        final float x1, y1, x2, y2;

        public Rect(float x1, float y1, float x2, float y2) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
        }
    }

    public static class Point {
        final float x, y;

        public Point(float x, float y) {
            this.x = x;
            this.y = y;
        }
    }
}
