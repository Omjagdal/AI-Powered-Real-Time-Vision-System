#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace dnn;
using namespace std;

int main() {

    // Load YOLO model
    Net net = readNet(
        "yolov3-tiny.weights",
        "yolov3-tiny.cfg"
    );

    // Check model
    if (net.empty()) {
        cout << "YOLO model failed to load!" << endl;
        return -1;
    }

    cout << "YOLO model loaded successfully!" << endl;

    // Load class names
    vector<string> classes;

    ifstream file("coco.names");

    string line;

    while (getline(file, line)) {
        classes.push_back(line);
    }

    cout << "Classes loaded: " << classes.size() << endl;

    // Open webcam
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cout << "Camera not opened!" << endl;
        return -1;
    }

    Mat frame;

    while (true) {

        // Capture frame
        cap.read(frame);

        if (frame.empty()) {
            break;
        }

        // Create blob
        Mat blob;

        blobFromImage(
            frame,
            blob,
            1 / 255.0,
            Size(416, 416),
            Scalar(0, 0, 0),
            true,
            false
        );

        // Set input
        net.setInput(blob);

        // Forward pass
        vector<Mat> outputs;

        vector<String> outNames =
            net.getUnconnectedOutLayersNames();

        net.forward(outputs, outNames);

        // Process detections
        for (const auto& output : outputs) {

            float* data = (float*)output.data;

            for (int i = 0; i < output.rows; ++i, data += output.cols) {

                // Objectness score
                float objectness = data[4];

                // Class scores
                Mat scores = output.row(i).colRange(5, output.cols);

                Point classIdPoint;

                double classScore;

                minMaxLoc(
                    scores,
                    0,
                    &classScore,
                    0,
                    &classIdPoint
                );

                // Final confidence
                double confidence =
                    objectness * classScore;

                // Detection threshold
                if (confidence > 0.3) {

                    int centerX =
                        (int)(data[0] * frame.cols);

                    int centerY =
                        (int)(data[1] * frame.rows);

                    int width =
                        (int)(data[2] * frame.cols);

                    int height =
                        (int)(data[3] * frame.rows);

                    int left =
                        centerX - width / 2;

                    int top =
                        centerY - height / 2;

                    // Boundary checks
                    left = max(0, left);
                    top = max(0, top);

                    width =
                        min(width, frame.cols - left);

                    height =
                        min(height, frame.rows - top);

                    // Draw bounding box
                    rectangle(
                        frame,
                        Rect(left, top, width, height),
                        Scalar(0, 255, 0),
                        2
                    );

                    // Create label
                    string label =
                        classes[classIdPoint.x] +
                        " " +
                        to_string((int)(confidence * 100)) + "%";

                    // Draw label
                    putText(
                        frame,
                        label,
                        Point(left, top - 10),
                        FONT_HERSHEY_SIMPLEX,
                        0.5,
                        Scalar(0, 255, 0),
                        2
                    );
                }
            }
        }

        // Show output
        imshow(
            "VisionBoost - Real-Time Object Detection",
            frame
        );

        // Exit on ESC
        if (waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}