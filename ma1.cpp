#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;

int main()
{
    Net net = readNetFromONNX("yolo11n.onnx");

    if(net.empty())
    {
        std::cout << "Failed to load ONNX model\n";
        return -1;
    }

    std::cout << "ONNX model loaded successfully\n";

    return 0;
}