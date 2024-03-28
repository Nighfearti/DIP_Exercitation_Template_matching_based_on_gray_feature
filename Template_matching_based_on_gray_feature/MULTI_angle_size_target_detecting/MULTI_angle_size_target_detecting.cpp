#pragma warning(disable:4996)
#include<iostream>
#include <opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>   
#include "opencv2/imgproc/imgproc_c.h" 
#include<vector>
#include<filesystem>

using std::vector;
using namespace std;
using namespace cv;
void multFind(Mat img1, Mat img2, Mat img3, double& best_angle_out, double& best_zoom, Point& best_match_zoom_and_angle, double& max_match, double k,string save_path);


//计算图像特征向量
//templ为模板，prototxtPath和caffemodelPath分别为两个文件的路径
cv::Mat extractResNet50Features(const cv::Mat& templ, const std::string& prototxtPath, const std::string& caffemodelPath) {
	cv::dnn::Net net = cv::dnn::readNetFromCaffe(prototxtPath, caffemodelPath);
	cv::Mat blob = cv::dnn::blobFromImage(templ, 1.0, cv::Size(224, 224), cv::Scalar(103.93, 116.77, 123.68));
	net.setInput(blob);
	cv::Mat feature = net.forward("fc1000");
	return feature;
}

//特征向量距离比对，并在待测图像上标出所有符合要求的区域
//feature为模板特征向量， featuresOfLabels为所有疑似符合要求的区域的特征向量的集合，img为原始图像
void compareAndMarkMatches(const cv::Mat& feature, const std::vector<cv::Mat>& featuresOfLabels, cv::Mat& img, const cv::Mat& labels, const cv::Mat& templ, int numComponents) {
	for (int i = 1; i < numComponents; i++) {
		double dist = cv::norm(feature, featuresOfLabels[i], cv::NORM_L2);
		cv::Mat points;
		cv::findNonZero(labels == i, points);
		cv::Rect roiRect = cv::boundingRect(points);
		if (dist < 0.01) {
			roiRect.width = templ.cols;
			roiRect.height = templ.rows;
			cv::rectangle(img, roiRect, cv::Scalar(0, 0, 255), 2);
		}
	}
}



//struct Multisubject{
//	double  Val;//匹配度
//	Point  Loc;//框的左上点
//	int delta_x;//横向框宽
//	int delta_y;//纵向框高
//	};
//
//Mat RotateImg(Mat image, double angle)
//{
//	/*
//	对旋转的进行改进，由于图形是一个矩形，旋转后的新图像的形状是一个原图像的外接矩形
//	因此需要重新计算出旋转后的图形的宽和高
//	*/
//	int width = image.cols;
//	int height = image.rows;
//
//	double radian = angle * CV_PI / 180.;//角度转换为弧度
//	double width_rotate = fabs(width * cos(radian)) + fabs(height * sin(radian));
//	double height_rotate = fabs(width * sin(radian)) + fabs(height * cos(radian));
//
//	//旋转中心 原图像中心点
//	cv::Point2f center((float)width / 2.0, (float)height / 2.0);
//	//旋转矩阵
//	Mat m1 = cv::getRotationMatrix2D(center, angle, 1.0);
//	//m1为2行3列通道数为1的矩阵
//	//变换矩阵的中心点相当于平移一样 原图像的中心点与新图像的中心点的相对位置
//	m1.at<double>(0, 2) += (width_rotate - width) / 2.;
//	m1.at<double>(1, 2) += (height_rotate - height) / 2.;
//	Mat imgOut;
//	if (image.channels() == 1)
//	{
//		cv::warpAffine(image, imgOut, m1, cv::Size(width_rotate, height_rotate), cv::INTER_LINEAR, 0, Scalar(255));
//	}
//	else if (image.channels() == 3)
//	{
//		cv::warpAffine(image, imgOut, m1, cv::Size(width_rotate, height_rotate), cv::INTER_LINEAR, 0, Scalar(255, 255, 255));
//	}
//	return imgOut;
//}

//void string_replace(std::string& strBig, const std::string& strsrc, const std::string& strdst)
//{
//	std::string::size_type pos = 0;
//	std::string::size_type srclen = strsrc.size();
//	std::string::size_type dstlen = strdst.size();
//
//	while ((pos = strBig.find(strsrc, pos)) != std::string::npos)
//	{
//		strBig.replace(pos, srclen, strdst);
//		pos += dstlen;
//	}
//}

//std::string GetPathOrURLShortName(std::string strFullName)
//{
//	if (strFullName.empty())
//	{
//		return "";
//	}
//
//	string_replace(strFullName, "/", "\\");
//
//	std::string::size_type iPos = strFullName.find_last_of('\\') + 1;
//
//	return strFullName.substr(iPos, strFullName.length() - iPos);
//}

//void findBest(Mat img1, Mat img2,Mat img3, double& best_angle_out, double& best_zoom, Point& best_match_zoom_and_angle, double& max_match, double k);
//
//void findSecond(Mat img1, Mat img2, Mat img3, double& best_angle_out, double& best_zoom, Point& best_match_zoom_and_angle, double& max_match, double Biggest, double k, int n);


//void show(Mat img1, Mat img2, double& angle, Point& Loc);


int main() {
	//string query = "airplane56.tif";
	//string templ = "5.png";
	//string strFileName = GetPathOrURLShortName(templ);

		//此为选项指标，每次均观察其值
	String flagtemp = "";
	String flagquery = "";

	//图像加载界面
	String pathtemp = "plane.png";
	String pathquery = "airport.tif";
	//cout << "请输入你\033[32m 模板\033[0m 的 \033[32m绝对地址\033[0m \n\n\033[32m 注意，路径中的文件夹不能有空格 \033[0m\n\n 如果输入数字 \033[32m 0 \033[0m ，将会使用默认模板" << endl << endl << ">> " << flush;
	//cin >> flagtemp;
	//cout << '\n' << '\n';
	//cout << "请输入你要操作的\033[32m 待匹配图像\033[0m 的 \033[32m 绝对地址 \033[0m \n\n\033[32m 注意，路径中的文件夹不能有空格 \033[0m\n\n如果输入数字 \033[32m 0 \033[0m ，将会使用默认待匹配图像" << endl << endl << ">> " << flush;
	//cin >> flagquery;
	//cout << '\n' << '\n';
		// 输入模板地址
	do {
		std::cout << "请输入你\033[32m 模板\033[0m 的 \033[32m绝对地址\033[0m \n\n\033[32m 注意，路径中的文件夹不能有空格,否则重新输入 \033[0m\n\n 如果输入数字 \033[32m 0 \033[0m ，将会使用默认模板" << endl << endl << ">> " << flush;
		std::cin >> flagtemp;

		// 在这里添加你的验证逻辑，例如检查文件是否存在等
		// 如果 flagtemp 不符合要求，可以输出错误消息并继续循环

	} while (flagtemp != "0" || flagtemp.empty());  // 继续循环直到用户输入有效值

	std::cout << '\n' << '\n';

	// 输入待匹配图像地址
	do {
		std::cout << "请输入你要操作的\033[32m 待匹配图像\033[0m 的 \033[32m 绝对地址 \033[0m \n\n\033[32m 注意，路径中的文件夹不能有空格,否则重新输入 \033[0m\n\n如果输入数字 \033[32m 0 \033[0m ，将会使用默认待匹配图像" << endl << endl << ">> " << flush;
		std::cin >> flagquery;

		// 在这里添加你的验证逻辑，例如检查文件是否存在等
		// 如果 flagquery 不符合要求，可以输出错误消息并继续循环

	} while (flagquery != "0" || flagquery.empty());  // 继续循环直到用户输入有效值

	std::cout << '\n' << '\n';
	flagtemp == "0" ? pathtemp : pathtemp = flagtemp;
	flagquery == "0" ? pathquery : pathquery = flagquery;


	Mat img2 = imread(pathtemp);
	if (img2.empty()) {
		std::cerr << "无法加载图像" << pathtemp << std::endl;
		return -1;
	}
	cout << endl << "模板打开成功，图像路径是" << pathtemp << endl << endl;
	//cout << '\n' << '\n';

	Mat img1 = imread(pathquery);
	if (img1.empty()) {
		cerr << "无法加载图像" << pathquery << std::endl;
		return -1;
	}


	cout << endl << "待匹配图像打开成功，图像路径是" << pathquery << endl << endl;
	cout << '\n' << '\n';

	//过渡缓冲
	system("pause");
	system("cls");


	double thresholdvalue = 0.6;  // 最佳匹配值

	std::string bowl;
again:
	std::cout << "请输入你想选择的 \033[32m匹配阈值（0到1）\033[0m，如果输入 \033[32m123\033[0m，将会使用默认匹配阈值" << std::endl << std::endl << ">> " << std::flush;
	std::cin >> bowl;
	std::cout << '\n';

	if (bowl == "123") {
		// 保持默认值
	}
	else {
		try {
			double input_value = std::stod(bowl);
			// 确保输入在有效范围内
			if (input_value >= 0.0 && input_value <= 1.0) {
				thresholdvalue = input_value;
			}
			else {
				std::cerr << "错误：输入值不在有效范围内（0到1）,请重新输入：" << '\n' << std::endl;
				goto again;
			}
		}
		catch (const std::invalid_argument& e) {
			std::cerr << "错误：无效的输入（不是一个数字）,请重新输入：" << '\n' << std::endl;
			goto again;

		}
		catch (const std::out_of_range& e) {
			std::cerr << "错误：输入超出范围,请重新输入：" << '\n' << std::endl;
			goto again;
		}
	}

	cout << endl << "设置成功，匹配阈值是" << thresholdvalue << endl << endl; cout << '\n' << '\n';

	string savepath;
	while (true) {
		std::cout << "\t请输入您想保存图片的 \033[32m文件夹的绝对路径\033[0m \n\t注意，不能输入 \033[32m不存在的文件夹\033[0m \n\t如果输入 \033[32mdefault\033[0m ，则将图片保存在默认文件夹\033[32m'../savepics'\033[0m中：" << std::endl;
		std::cout<<'\n' << ">> ";
		std::cin >> savepath;
		std::cout << '\n'<<'\n';

		if (savepath == "default") {
			savepath = "../savepics";
			break;  // 跳出循环
		}

		// 检查文件夹是否存在
		if (std::filesystem::exists(savepath)) {
			break;  // 跳出循环
		}
		else {
			std::cerr << "\t错误：输入的文件夹路径不存在。请重新输入。" << std::endl;
		}
	}

	std::cout << "\t\t图片将保存在文件夹：\"" << savepath <<"\"" << std::endl<<'\n';

	// 假设你已经知道完整的保存路径 save_path
	//std::string save_path = "../savepics/saved_image.jpg";  // 请替换为你想要的完整路径和文件名
	 // 提取文件名（不包括扩展名）
	std::filesystem::path temp_path(pathtemp);
	std::filesystem::path query_path(pathquery);
	std::string temp_filename = temp_path.stem().string();
	std::string query_filename = query_path.stem().string();

	// 构建保存路径
	std::string save_path = savepath + "/" + temp_filename + "_in_" + query_filename + ".jpg";



	//过渡缓冲
	system("pause");
	system("cls");

	imshow("Query", img1);
	imshow("Template", img2);
	Mat img3 = img1.clone();
	double bestMatchValue = 0.0;
	Point bestMatchLoc;           // 最佳匹配位置
	double bestMatchAngle = 0.0;  // 最佳匹配的旋转角度
	double bestMatchScale = 0.0;  // 最佳伸缩大小

	multFind(img1, img2, img3, bestMatchAngle, bestMatchScale, bestMatchLoc, bestMatchValue, thresholdvalue, save_path);
	///
	/// 
	///

	waitKey(3000);

	return 0;
}

void saveImage(const cv::Mat& img, const std::string& save_path) {
	if (img.empty()) {
		std::cerr << "图像为空，无法保存" << std::endl;
		return;
	}

	// 将图像写入文件
	if (cv::imwrite(save_path, img)) {
		std::cout << "图像成功保存到文件：\"" << save_path <<"\"" << std::endl<<'\n';
	}
	else {
		std::cerr << "保存图像失败" << std::endl;
	}
}

Mat rotateImage(const Mat& source, double angle)
{
	//旋转函数
	Point2f src_center(source.cols / 2.0F, source.rows / 2.0F);//图像中心点
	Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);//生成旋转矩阵，缩放因子为1（无缩放）
	Mat dst;
	warpAffine(source, dst, rot_mat, source.size());//使用旋转矩阵对原图像进行仿射变换
	return dst;
}

Mat zoomImage(const Mat& source, double ratio) {
	Mat zoom_mat;
	resize(source, zoom_mat, Size(), ratio, ratio, INTER_CUBIC);
	return zoom_mat;
}


void findBest(Mat img1, Mat img2, Mat img3, double& best_angle_out, double& best_zoom, Point& best_loc, double& max_match, double k) {

	double max_match_zoom_and_angle = -1;

	// 对不同角度的模板图像进行匹配
	for (double zoomratio = 0.5; zoomratio <= 1.5; zoomratio += 0.1)//尝试不同的缩放比例
	{
		Mat zoomed = zoomImage(img2, zoomratio);//进行缩放

		// 对不同角度的模板图像进行匹配
		Point best_match;
		double best_angle = 0;

		//在某一缩放比例下尝试不同的角度
		for (double angle = 0; angle < 360; angle += 10)
		{
			Mat rotated = rotateImage(zoomed, angle);//旋转角度的模板
			Mat dstImg;
			matchTemplate(img1, rotated, dstImg, TM_CCOEFF_NORMED);//将匹配数据存至dstImg

			double  maxVal;
			Point  maxLoc;
			minMaxLoc(dstImg, 0, &maxVal, 0, &maxLoc);//取出最匹配处的坐标

			if (maxVal > max_match_zoom_and_angle)//检查当前的最大匹配值是否大于迄今为止找到的最大匹配值
			{
				max_match_zoom_and_angle = maxVal;//匹配值
				best_match = maxLoc;//匹配坐标
				best_angle = angle;//匹配角
			}
		}

		if (max_match_zoom_and_angle > max_match)//检查当前的最大匹配值是否大于迄今为止找到的最大匹配值
		{
			max_match = max_match_zoom_and_angle;//匹配值
			best_loc = best_match;//匹配坐标
			best_zoom = zoomratio;//匹配放缩程度
			best_angle_out = best_angle;//当前旋转
		}
	}
	// 绘制最佳匹配结果
	Mat zoomed_best = zoomImage(img2, best_zoom);
	Mat best = rotateImage(zoomed_best, best_angle_out);

	//imshow("rotate",best);

	int x = best.cols / 2;    // x 对应列坐标
	int y = best.rows / 2;    // y 对应行坐标
	int width = best.cols;
	int height = best.rows;
	cv::Rect rect(best_loc.x, best_loc.y, width, height);

	//将矩形贴到img中，并将矩形区域置为黑色
	cv::Mat subImg = img1(rect);
	subImg.setTo(0);

	if (max_match < k) {
		std::cout << "没有符合阈值的图像，5秒后退出" << '\n' << '\n';

		std::this_thread::sleep_for(std::chrono::seconds(5));
		// 退出程序
		exit(0);
	}

	RotatedRect rotatedRect(cv::Point2f(best_loc.x + best.cols / 2, best_loc.y + best.rows / 2), best.size(), best_angle_out);
	/*ellipse(img3, rotatedRect, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);*/
	  // 获取 RotatedRect 的四个顶点
	cv::Point2f vertices[4];
	rotatedRect.points(vertices);

	// 在图像上绘制旋转矩形
	for (int i = 0; i < 4; ++i) {
		cv::line(img3, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
	}

	string title = "Best:";
	std::string max_match_str = std::to_string(max_match);
	size_t dot_pos = max_match_str.find('.');
	if (dot_pos != std::string::npos && max_match_str.length() > dot_pos + 4) {
		max_match_str = max_match_str.substr(0, dot_pos + 4);
	}
	//rectangle(img3, best_loc, Point(best_loc.x + best.cols, best_loc.y + best.rows), Scalar(0, 255, 0), 2, 8);
	putText(img3, title, cv::Point2f(best_loc.x, best_loc.y + best.rows / 6), FONT_HERSHEY_PLAIN, 0.8, Scalar(0, 255, 0), 0.8, LINE_8);
	putText(img3, max_match_str, cv::Point2f(best_loc.x, best_loc.y + best.rows / 6 + 16), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255), 0.8, LINE_8);
	imshow("Best zoom-and-rotated ", img3);
	system("cls");
	cout << "max_value=" << max_match_zoom_and_angle << endl;
	cout << "best_location" << best_loc << endl;
	cout << "best_zoom=" << best_zoom << endl;
	cout << "best_angle=" << best_angle_out << endl<<'\n';

	waitKey(100);
}

void findSecond(Mat img1, Mat img2, Mat img3, double& best_angle_out, double& best_zoom, Point& best_loc, double& max_match, double Biggest, double k, int n) {

	double max_match_zoom_and_angle = -1;

	// 对不同角度的模板图像进行匹配
	for (double zoomratio = 0.5; zoomratio <= 1.5; zoomratio += 0.1)//尝试不同的缩放比例
	{
		Mat zoomed = zoomImage(img2, zoomratio);//进行缩放

		// 对不同角度的模板图像进行匹配
		Point best_match;
		double best_angle = 0;

		//在某一缩放比例下尝试不同的角度
		for (double angle = 0; angle < 360; angle += 10)
		{
			Mat rotated = rotateImage(zoomed, angle);//旋转角度的模板
			Mat dstImg;
			matchTemplate(img1, rotated, dstImg, TM_CCOEFF_NORMED);//将匹配数据存至dstImg

			double  maxVal;
			Point  maxLoc;
			minMaxLoc(dstImg, 0, &maxVal, 0, &maxLoc);//取出最匹配处的坐标

			if (maxVal > max_match_zoom_and_angle)//检查当前的最大匹配值是否大于迄今为止找到的最大匹配值
			{
				max_match_zoom_and_angle = maxVal;//匹配值
				best_match = maxLoc;//匹配坐标
				best_angle = angle;//匹配角
			}
		}

		if (max_match_zoom_and_angle > max_match && (max_match_zoom_and_angle < Biggest))//检查当前的最大匹配值是否大于迄今为止找到的最大匹配值
		{
			max_match = max_match_zoom_and_angle;//匹配值
			best_loc = best_match;//匹配坐标
			best_zoom = zoomratio;//匹配放缩程度
			best_angle_out = best_angle;//当前旋转
		}
	}

	// 绘制最佳匹配结果
	Mat zoomed_best = zoomImage(img2, best_zoom);
	Mat best = rotateImage(zoomed_best, best_angle_out);
	//imshow("rotate", best);

	int x = best.cols / 2;    // x 对应列坐标
	int y = best.rows / 2;    // y 对应行坐标
	int width = best.cols;
	int height = best.rows;
	cv::Rect rect(best_loc.x, best_loc.y, width, height);


	if (max_match < k) {
		//// 退出函数
		return;
	}

	//将矩形贴到img中，并将矩形区域置为黑色
	cv::Mat subImg = img1(rect);
	subImg.setTo(0);



	RotatedRect rotatedRect(cv::Point2f(best_loc.x + best.cols / 2, best_loc.y + best.rows / 2), best.size(), best_angle_out);
	/*ellipse(img3, rotatedRect, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);*/
	  // 获取 RotatedRect 的四个顶点
	cv::Point2f vertices[4];
	rotatedRect.points(vertices);


	// 在图像上绘制旋转矩形
	for (int i = 0; i < 4; ++i) {
		cv::line(img3, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
	}

	string title = "Good:";
	std::string max_match_str = std::to_string(max_match);
	size_t dot_pos = max_match_str.find('.');
	if (dot_pos != std::string::npos && max_match_str.length() > dot_pos + 4) {
		max_match_str = max_match_str.substr(0, dot_pos + 4);
	}
	//rectangle(img3, best_loc, Point(best_loc.x + best.cols, best_loc.y + best.rows), Scalar(0, 255, 0), 2, 8);
	putText(img3, title, cv::Point2f(best_loc.x, best_loc.y + best.rows / 6), FONT_HERSHEY_PLAIN, 0.8, Scalar(0, 0, 255), 0.8, LINE_8);
	putText(img3, max_match_str, cv::Point2f(best_loc.x, best_loc.y + best.rows / 6 + 16), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0), 0.8, LINE_8);
	//rectangle(img3, best_loc, Point(best_loc.x + best.cols, best_loc.y + best.rows), Scalar(0, 255, 0), 2, 8);
	//putText(img1, strFileName, cv::Point(best_match_zoom_and_angle.x, best_match_zoom_and_angle.y + 20), FONT_HERSHEY_PLAIN, 1.2, Scalar(0, 0, 255), 1.2, LINE_8);
	imshow("Best zoom-and-rotated ", img3);
	cout << "max_value=" << max_match << endl;
	cout << "best_location" << best_loc << endl;
	cout << "best_zoom=" << best_zoom << endl;
	cout << "best_angle=" << best_angle_out << endl<<'\n';

	waitKey(100);
}

void multFind(Mat img1, Mat img2, Mat img3, double& best_angle_out, double& best_zoom, Point& best_match_zoom_and_angle, double& max_match, double k,string save_path) {
	findBest(img1, img2, img3, best_angle_out, best_zoom, best_match_zoom_and_angle, max_match, k);
	//show(img1, img2, angle, Loc);
	double newVal = k + 1;
	int n = 0;
	if (max_match > k)n++;
	do
	{
		newVal = 0;
		findSecond(img1, img2, img3, best_angle_out, best_zoom, best_match_zoom_and_angle, newVal, max_match, k, n);
		//show(img1, img2, angle, Loc);
		max_match = newVal;
		if (max_match > k)n++;
	} while (newVal > k);
	cout << '\n';
	std::cout << "共有" << n << "个符合阈值的图像" << '\n' << '\n';
	//// 假设你已经知道完整的保存路径 save_path
	//std::string save_path = "../savepics/saved_image.jpg";  // 请替换为你想要的完整路径和文件名
	// 调用保存函数
	saveImage(img3, save_path);
	//cv::imwrite(save_path, img3);
	// 输出提示信息
	std::cout << "文件已保存至:\"" << save_path << "\",5秒后退出" << std::endl;

	std::this_thread::sleep_for(std::chrono::seconds(5));
}


//void show(Mat img1, Mat img2, double& angle, Point& Loc) {
//	Mat rotated_best = rotateImage(img2, angle);
//	rectangle(img1, Loc, Point(Loc.x + rotated_best.cols, Loc.y + rotated_best.rows), Scalar(0, 255, 0), 2, 8);
//	imshow("Best Match", img1);
//	waitKey(0);
//}
