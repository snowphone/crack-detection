/*
 * 컴파일시 C:\ 에 opencv 라이브러리를 갖다놓을 것.
 * 적용 결과 adpative thresholding은 효과가 좋지 않으므로,
 * 기본적인 global thresholding + otsu's binarization(경계값을 자동으로 추출)을 사용한다.
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

#include <opencv2/opencv.hpp>

using namespace std;
namespace fs = std::experimental::filesystem;

void convertToBinary(cv::Mat& image);
void rotate(cv::Mat image, const int degree);
void showImage(const cv::Mat& image, double ratio);

int main(int argc, char** argv)
{
	auto path = fs::path(argv[1]).parent_path(),
		colorPath = fs::path(path.generic_string() + "/color"),
		bwPath = fs::path(path.generic_string() + "/bw");

	fs::create_directory(colorPath);
	fs::create_directory(bwPath);

	for (size_t i = 1; i < argc; i++)
	{
		cv::Mat img = cv::imread(argv[i]);
		const string filename = fs::path(argv[i]).filename().generic_string();

		for (size_t degree = 0; degree < 360; degree += 90)
		{
			cv::Mat working;
			img.copyTo(working);
			rotate(working, degree);
			cv::imwrite(colorPath.generic_string() + '/' + to_string(degree) + "deg_" + filename, working);

			convertToBinary(working);
			cv::imwrite(bwPath.generic_string() + '/' + to_string(degree) + "deg_" + filename, working);

			printf("%llu/%d\t%s\n", i, argc, filename.c_str());
		}
	}
}

void showImage(const cv::Mat& image, double ratio)
{
	cv::namedWindow("window", cv::WINDOW_GUI_EXPANDED);
	cv::resizeWindow("window", image.cols * ratio, image.rows * ratio);
	cv::imshow("window", image);
	cv::waitKey();
}

//handling 0 degrees.
void rotate(cv::Mat image, const int degree)
{
	if (degree == 0)
		return;
	cv::rotate(image, image, degree / 90);
}

void convertToBinary(cv::Mat& image)
{
	cv::GaussianBlur(image, image, { 5, 5 }, 0);
	cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
	cv::threshold(image, image, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	//cv::threshold(image, image, 35, 255, cv::THRESH_BINARY );
}