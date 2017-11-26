#include <exception>
#include <ctime>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <random>
#include <chrono>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>

#include "GpuMat.cuh"
#include "GpuVector.cuh"
#include "helpers.h"

namespace fs = std::experimental::filesystem;

std::vector<std::vector<int>> getColours(int N)
{
	int max_value = 255-25;
	int min_value = 25;
	
	int perChannel = max_value - min_value;
	if(N > perChannel*perChannel*perChannel)
		throw "N must be less than the number of colours";

	std::vector<int> rV, gV, bV;
	for(int r = min_value; r <= max_value; r++)
		for(int g = min_value; g <= max_value; g++)
			for(int b = min_value; b <= max_value; b++)
			{
				rV.push_back(r);
				gV.push_back(g);
				bV.push_back(b);
			}
	
	std::random_device r;
	std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};

	// create two random engines with the same state
	std::mt19937 eng1(seed);
	auto eng2 = eng1;
	auto eng3 = eng1;

	std::shuffle(rV.begin(), rV.end(), eng1);
	std::shuffle(gV.begin(), gV.end(), eng2);
	std::shuffle(bV.begin(), bV.end(), eng3);

	std::vector<std::vector<int>> ret;
	ret.push_back(std::vector<int>(rV.begin(), rV.begin() + N));
	ret.push_back(std::vector<int>(gV.begin(), gV.begin() + N));
	ret.push_back(std::vector<int>(bV.begin(), bV.begin() + N));
	return ret;
}

/**
 * contains cuda specific initializations
 */
int main(int argc, char** argv )
{	
	// grab the arguments
	std::string image_path;
	for (int i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "-i") == 0)
			image_path = argv[i+1];
	}
	
	std::vector<std::string> all_files;

	for(auto& p: fs::directory_iterator(image_path))
	{
		std::string val = p.path().string();
		if(Helpers::hasEnding(val,"png"))
			all_files.push_back(val);
	}

	std::cout << "found " << all_files.size() << " images" << std::endl;

	cv::Mat background;
	cv::Mat covered;

	std::unique_ptr<GpuMat<float>> maxVals;
	std::unique_ptr<GpuMat<int>> maxInds;
	std::unique_ptr<GpuMat<float>> scratchGpuMat;

	float convertTime = 0.f;
	float readTime = 0.f;

	int maxInd = 0;

	std::vector<cv::Mat> all_affected;
	auto startTime = std::chrono::high_resolution_clock::now(); //to beat 58 s
	std::set<char> delims;
	delims.emplace('/');
	for(int i = 0; i < all_files.size(); i++)
	{
		std::vector<std::string> split = Helpers::splitpath(all_files[i],delims);
		std::string fileName = Helpers::remove_extension(split[split.size()-1]);
		int curInd = atoi(fileName.c_str());
		if(curInd > maxInd)
			maxInd = curInd;
		
		auto readStart = std::chrono::high_resolution_clock::now();
		cv::Mat img = cv::imread(all_files[i], CV_LOAD_IMAGE_COLOR);
		std::vector<cv::Mat> channels;
		cv::split(img, channels);
		auto readEnd = std::chrono::high_resolution_clock::now();

		auto convertStart = std::chrono::high_resolution_clock::now();
		cv::Mat temp;
		channels[0].convertTo(temp, CV_32FC1);
		auto convertEnd = std::chrono::high_resolution_clock::now();

		if(i == 0)
		{
			scratchGpuMat = std::unique_ptr<GpuMat<float>>(new GpuMat<float>(temp));//do this to allocate memory
			maxVals = std::unique_ptr<GpuMat<float>>(new GpuMat<float>(temp));
			maxInds = std::unique_ptr<GpuMat<int>>(new GpuMat<int>(maxVals->height, maxVals->width, maxVals->depth, curInd));
		}
		else
		{
			scratchGpuMat->load(temp);
			maxVals->maxTrackInds(*scratchGpuMat, maxInds.get(), curInd);
		}
		
		convertTime += (float)std::chrono::duration_cast<std::chrono::milliseconds>(convertEnd - convertStart).count()/1000.f;
		readTime += (float)std::chrono::duration_cast<std::chrono::milliseconds>(readEnd - readStart).count()/1000.f;
	}
	auto endTime = std::chrono::high_resolution_clock::now();

	float totalTime = (float)std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()/1000.0f;

	std::vector<std::vector<int>> colours = getColours(all_files.size());

	GpuVector<int> r(colours[0]), g(colours[1]), b(colours[2]);

	//cv::Mat inds = maxInds->getMat();

	// cv::Mat vals = maxVals->getMat();
	// vals.convertTo(vals, CV_8U);
	// cv::imshow("vals", vals);

	GpuMat<float> resGpu(maxInds->height, maxInds->width, 3, false);
	resGpu.indsToColour(*maxInds, *maxVals, r, g, b);

	cv::Mat res = resGpu.getMat();
	res.convertTo(res, CV_8UC3);

	//std::cout << colours.size() << ", " << colours[0].size() << ", " << colours[1].size() << ", " << colours[2].size() << std::endl;
	//std::cout << colours[0][10] << ", " << colours[1][10] << ", " << colours[2][10] << std::endl;

	cv::imshow("res", res);
	cv::waitKey(0);

	cv::imwrite("res.png", res);

	std::cout << std::endl;
	std::cout << "total time: " << totalTime << " seconds" << std::endl;
	std::cout << "convert time: " << convertTime << " seconds" << std::endl;
	std::cout << "read time: " << readTime << " seconds" << std::endl;

	return 0;
}
