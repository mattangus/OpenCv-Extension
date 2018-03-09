#include <exception>
#include <ctime>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <random>
#include <chrono>
#include <regex>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>

#include "GpuMat.cuh"
#include "GpuVector.cuh"
#include "helpers.h"

namespace fs = std::experimental::filesystem;

inline int cantorPair(int k1, int k2)
{
	return (k1 + k2)*(k1 + k2 + 1)/2 + k2;
}

int cantorPair(std::vector<int> vals)
{
	int cur = vals[0];
	for(int i = 1; i < vals.size(); i++)
	{
		cur = cantorPair(cur, vals[i]);
	}
	return cur;
}

std::vector<std::vector<int>> getColours(std::string fileName, int* maxVal)
{
	std::vector<std::vector<int>> ret;
	std::ifstream ifs(fileName);
	
	for(std::string line; std::getline(ifs, line); )
	{
		int r, g, b, id;
		std::istringstream in(line);

		in >> r >> g >> b >> id;
		int val = ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
		if(val > *maxVal)
			*maxVal = val;
		std::vector<int> temp = {val, id};
		ret.push_back(temp);
	}
	return ret;
}

/**
 * contains cuda specific initializations
 */
int main(int argc, char** argv )
{	
	// grab the arguments
	std::string image_path;
	std::string coloursFile;
	std::string output_path;
	for (int i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "-i") == 0)
			image_path = argv[i+1];
		if (strcmp(argv[i], "-o") == 0)
			output_path = argv[i+1];
		if (strcmp(argv[i], "-c") == 0)
			coloursFile = argv[i+1];
	}
	int maxCVal = -1000;
	auto colours = getColours(coloursFile, &maxCVal);
	std::cout.flush();
	std::vector<int> colourMap(maxCVal, 0);

	for(int i = 0; i < colours.size(); i++)
	{
		colourMap[colours[i][0]] = colours[i][1];
	}

	GpuVector<int> gpu_colourMap(colourMap);

	std::cout << "gpu colour map has " << gpu_colourMap.numElem << " elements" << std::endl;

	std::vector<std::string> all_files;

	for(auto& p: fs::recursive_directory_iterator(image_path))
	{
		std::string val = p.path().string();
		if(Helpers::hasEnding(val,"png"))
			all_files.push_back(val);
	}

	std::cout << "found " << all_files.size() << " images" << std::endl;

	std::unique_ptr<GpuMat<int>> scratchGpuMat;
	std::unique_ptr<GpuMat<int>> outputImg;

	float readTime = 0.f;

	auto startTime = std::chrono::high_resolution_clock::now(); //to beat 58 s
	for(int i = 0; i < all_files.size(); i++)
	{
		auto readStart = std::chrono::high_resolution_clock::now();
		std::cout << "reading " << all_files[i] << std::endl;
		cv::Mat img = cv::imread(all_files[i], CV_LOAD_IMAGE_COLOR);
		img.convertTo(img, CV_32SC3);
		auto readEnd = std::chrono::high_resolution_clock::now();

		if(i == 0)
		{
			scratchGpuMat = std::unique_ptr<GpuMat<int>>(new GpuMat<int>(img.rows, img.cols, img.channels(), false));//do this to allocate memory
			outputImg = std::unique_ptr<GpuMat<int>>(new GpuMat<int>(img.rows, img.cols, img.channels(), false));
		}
		scratchGpuMat->load(img);
		scratchGpuMat->mapColours(*outputImg, gpu_colourMap); //(GpuMat<dtype>& to, GpuVector<dtype>& map)
		
		cv::Mat outMat = outputImg->getMat();

		outMat.convertTo(outMat, CV_8UC3);		

		std::string outFileName = std::regex_replace(all_files[i], std::regex(image_path), output_path);
		fs::create_directories(fs::path(outFileName).parent_path());
		std::cout << "writing " << outFileName << std::endl;
		cv::imwrite(outFileName, outMat);

		readTime += (float)std::chrono::duration_cast<std::chrono::milliseconds>(readEnd - readStart).count()/1000.f;
	}
	auto endTime = std::chrono::high_resolution_clock::now();

	float totalTime = (float)std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()/1000.0f;

	std::cout << std::endl;
	std::cout << "total time: " << totalTime << " seconds" << std::endl;
	std::cout << "read time: " << readTime << " seconds" << std::endl;

	return 0;
}
