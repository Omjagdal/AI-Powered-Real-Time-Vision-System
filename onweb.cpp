#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main()
{
    // Load ONNX model
    Net net = readNetFromONNX("yolo11n.onnx");

    if (net.empty())
    {
        cout << "Failed to load ONNX model!" << endl;
        return -1;
    }

    cout << "YOLO11 ONNX loaded successfully!" << endl;

    // Load class names
    vector<string> classNames;
    ifstream ifs("coco.names");

    string line;
    while (getline(ifs, line))
    {
        classNames.push_back(line);
    }

    cout << "Loaded classes: "
         << classNames.size()
         << endl;

    // Open webcam
    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "Failed to open webcam!" << endl;
        return -1;
    }

    Mat frame;

    while (true)
    {
        int64 start = getTickCount();

        cap >> frame;

        if (frame.empty())
            break;

        // Create blob
        Mat blob;

        blobFromImage(
            frame,
            blob,
            1.0 / 255.0,
            Size(640, 640),
            Scalar(),
            true,
            false
        );

        // Run inference
        net.setInput(blob);

        Mat output = net.forward();

        // Output shape:
        // 1 x 84 x 8400

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        const int rows = 8400;
        const int dimensions = 84;

        float* data = (float*)output.data;

        for (int i = 0; i < rows; i++)
        {
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            float* classes_scores = data + 4;

            Mat scores(
                1,
                80,
                CV_32FC1,
                classes_scores
            );

            Point classIdPoint;
            double maxClassScore;

            minMaxLoc(
                scores,
                0,
                &maxClassScore,
                0,
                &classIdPoint
            );

            if (maxClassScore > 0.25)
            {
                int left =
                    int((x - 0.5f * w) *
                        frame.cols / 640.0f);

                int top =
                    int((y - 0.5f * h) *
                        frame.rows / 640.0f);

                int width =
                    int(w *
                        frame.cols / 640.0f);

                int height =
                    int(h *
                        frame.rows / 640.0f);

                boxes.push_back(
                    Rect(
                        left,
                        top,
                        width,
                        height
                    )
                );

                confidences.push_back(
                    (float)maxClassScore
                );

                classIds.push_back(
                    classIdPoint.x
                );
            }

            data += dimensions;
        }

        // Apply NMS
        vector<int> indices;

        NMSBoxes(
            boxes,
            confidences,
            0.25,
            0.45,
            indices
        );

        // Draw detections
        for (int idx : indices)
        {
            Rect box = boxes[idx];

            rectangle(
                frame,
                box,
                Scalar(0, 255, 0),
                2
            );

            string label =
                classNames[classIds[idx]] +
                " " +
                format(
                    "%.2f",
                    confidences[idx]
                );

            putText(
                frame,
                label,
                Point(
                    box.x,
                    box.y - 10
                ),
                FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar(0, 255, 0),
                2
            );
        }

        // FPS
        double time =
            (getTickCount() - start) /
            getTickFrequency();

        double fps = 1.0 / time;

        putText(
            frame,
            "FPS: " + to_string((int)fps),
            Point(20, 40),
            FONT_HERSHEY_SIMPLEX,
            1,
            Scalar(0, 0, 255),
            2
        );

        imshow(
            "VisionBoost YOLO11 C++",
            frame
        );

        if (waitKey(1) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}