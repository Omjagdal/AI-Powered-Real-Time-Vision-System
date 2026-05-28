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
    int64 startTime;

    while (true) {

        startTime = cv::getTickCount();

        // Capture frame
        cap.read(frame);

        if (frame.empty()) {
            break;
        }

        // Convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Gaussian Blur
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

        // Canny Edge Detection
        cv::Canny(blurred, edges, 50, 150);

        // FPS Calculation
        double time = (cv::getTickCount() - startTime) / cv::getTickFrequency();

        fps = 1.0 / time;

        // Display FPS on frame
        cv::putText(
            edges,
            "FPS: " + std::to_string((int)fps),
            cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(255),
            2
        );

        // Show output
        cv::imshow("VisionBoost - Edge Detection", edges);

        // Exit on ESC
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}