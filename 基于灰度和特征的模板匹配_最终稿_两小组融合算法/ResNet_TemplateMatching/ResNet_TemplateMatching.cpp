// ResNet_TemplateMatching.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;

Mat rotate(Mat src, double angle)    // 旋转函数
{
    Mat dst;
    Point2f pt(src.cols / 2., src.rows / 2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, r, Size(src.cols, src.rows));
    return dst;
}

Mat zoomImage(Mat src, double zoomratio) // 缩放函数
{
    Mat dst;//创建承载缩放结果图像的Mat变量
    resize(src, dst, Size(), zoomratio, zoomratio);
    ///opencv 提供五种方法供选择分别是：
    ///a.最近邻插值——INTER_NEAREST；
    /// b.线性插值   ——INTER_LINEAR；（默认值）
    /// c.区域插值   ——INTER_AREA；(利用像素区域关系的重采样插值)
     /// d.三次样条插值——INTER_CUBIC（超过4 * 4像素邻域内的双三次插值）
    /// e.Lanczos插值——INTER_LANCZOS4（超过8 * 8像素邻域的Lanczos插值）
    return dst;
}

int main() {
    int Template_number = 0;
    double max_match = -1;//当前最好的匹配值
    Point best_loc;//当前最好的匹配位置
    double best_zoom = 1;//当前最好的缩放比例
    double best_angle_out = 0;//当前最好的旋转角度，是与缩放比例相匹配的值

    // 存储已找到的匹配区域
    std::vector<cv::Rect> rois;

    // 加载输入图像和模板
    cv::Mat img = cv::imread("airport.tif", cv::IMREAD_COLOR);
    cv::Mat templ = cv::imread("plane.png", cv::IMREAD_COLOR);

    for (double zoomratio = 0.6; zoomratio <= 0.8; zoomratio += 0.1) {  // 每次缩放0.1，从模板的0.6倍开始，一直到0.8
        Mat zoomedTempl = zoomImage(templ, zoomratio);  // 缩放模板

        for (int angle = 0; angle <= 180; angle += 90) {  // 每10度旋转一次
            Mat rotatedTempl = rotate(zoomedTempl, angle);  // 旋转模板      //此处使用的是之前受过缩放处理的zoomedTempl

            // 进行模板匹配
            cv::Mat result;
            cv::matchTemplate(img, rotatedTempl, result, cv::TM_CCOEFF_NORMED);

            // 归一化结果
            cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

            // 设置阈值
            double threshold = 0.95;

            // 找到所有可能的匹配区域
            cv::Mat mask;
            cv::threshold(result, mask, threshold, 1.0, cv::THRESH_BINARY);//二值化

            // 加载预训练的ResNet50模型
            //cv::dnn::Net net = cv::dnn::readNetFromCaffe("C:/Users/Chengshijia/Desktop/数字图像实习小组作业/GC_Templete_Matching/ResNet-50-deploy.prototxt", "C:/Users/Chengshijia/Desktop/数字图像实习小组作业/GC_Templete_Matching/ResNet-50-model.caffemodel");
            cv::dnn::Net net = cv::dnn::readNetFromCaffe("ResNet-50-deploy.prototxt", "ResNet-50-model.caffemodel");
            // 将模板预处理为ResNet50的输入格式
            cv::Mat blob = cv::dnn::blobFromImage(rotatedTempl, 1.0, cv::Size(224, 224), cv::Scalar(103.93, 116.77, 123.68));

            // 使用ResNet50提取模板的特征
            net.setInput(blob);
            cv::Mat feature = net.forward("fc1000");

            // 对每个可能的匹配区域提取特征并计算距离
            cv::Mat labels = cv::Mat::zeros(mask.size(), CV_32S); // 存储mask的连通区域
            mask.convertTo(mask, CV_8U);
            int numComponents = cv::connectedComponents(mask, labels);

            // 在循环外部计算可能的匹配区域的特征向量
            std::vector<cv::Mat> featuresOfLabels(numComponents);
            for (int i = 1; i < numComponents; i++) {
                cv::Mat maskOfLabel = cv::Mat::zeros(labels.size(), CV_8U);
                maskOfLabel.setTo(255, labels == i);  // 使用255而不是1，确保是单通道图像

                // 将 maskOfLabel 调整为与 templ 具有相同的大小
                cv::resize(maskOfLabel, maskOfLabel, rotatedTempl.size());

                cv::Mat roiOfTempl;
                rotatedTempl.copyTo(roiOfTempl, maskOfLabel);  // 提取标签区域的模板

                // 使用 ResNet-50 提取标签区域的特征向量
                cv::Mat blobOfLabel = cv::dnn::blobFromImage(roiOfTempl, 1.0, cv::Size(224, 224), cv::Scalar(103.93, 116.77, 123.68));
                net.setInput(blobOfLabel);
                featuresOfLabels[i] = net.forward("fc1000");
            }

            // 在循环内部进行比较
           // 在循环内部进行比较
            for (int i = 1; i < numComponents; i++) {
                // 与模板的特征向量进行比较
                double dist = cv::norm(feature, featuresOfLabels[i], cv::NORM_L2);

                // 计算标签区域的边界矩形
                cv::Mat points;
                cv::findNonZero(labels == i, points);
                cv::Rect roiRect = cv::boundingRect(points);

                // 如果距离小于阈值，则在原图上标记匹配区域
                if (dist < 0.01) {
                    // 提取匹配区域
                    roiRect.width = rotatedTempl.cols;
                    roiRect.height = rotatedTempl.rows;

                    // 检查新的匹配区域是否与已经找到的匹配区域重叠
                    bool overlap = false;
                    for (auto& existingRoi : rois) {
                        if ((roiRect & existingRoi).area() > 0) {
                            overlap = true;
                            if (dist > max_match) {
                                existingRoi = roiRect;
                            }
                            break;
                        }
                    }

                    if (!overlap) {
                        rois.push_back(roiRect);//保存匹配数据
                    }

                    // 检查当前的最大匹配值是否大于迄今为止找到的最大匹配值
                    if (dist > max_match) {//将匹配区域按照匹配度降序标记出来
                        max_match = dist;
                        best_loc = roiRect.tl();
                        best_zoom = zoomratio;
                        best_angle_out = angle;//将数据外传
                    }
                }
            }
        }
    }

    // 在原图上绘制所有的匹配区域
    for (const auto& roi : rois) {
        cv::rectangle(img, roi, cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("Match", img);
    cv::waitKey(0);
    return 0;
}



// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
