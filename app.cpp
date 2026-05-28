#include <opencv2/opencv.hpp>
#include <iostream>

int main() {

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Camera not opened!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat edges;

    double fps = 0;

    while (true) {

        int64 startTime = cv::getTickCount();

        // Capture frame
        cap.read(frame);

        if (frame.empty()) {
            break;
        }

        // Resize frame for dashboard
        cv::resize(frame, frame, cv::Size(640, 480));

        // Grayscale conversion
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Gaussian blur
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

        // Canny edge detection
        cv::Canny(blurred, edges, 50, 150);

        // FPS calculation
        double time =
            (cv::getTickCount() - startTime) /
            cv::getTickFrequency();

        fps = 1.0 / time;

        // Display FPS on original frame
        cv::putText(
            frame,
            "FPS: " + std::to_string((int)fps),
            cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(0, 255, 0),
            2
        );

        // Create windows
        cv::imshow("Original Frame", frame);
        cv::imshow("Grayscale Frame", gray);
        cv::imshow("Blurred Frame", blurred);
        cv::imshow("Edge Detection Frame", edges);

        // Exit on ESC
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}