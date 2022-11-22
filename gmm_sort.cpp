#include <iostream> 
#include "opencv2/opencv.hpp"  
#include "opencv2/video/background_segm.hpp"
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  

using namespace std;
using namespace cv;

vector<Point> mousev, kalmanv;
KalmanFilter KF;
Mat_<float> measurement(2, 1);   //模板矩阵
Mat_<float> state(4, 1); // (x, y, Vx, Vy)  
int incr = 0;
string num2str(int i)
{
	stringstream ss;
	ss << i;
	return ss.str();
}

void initKalman(float x, float y)
{ 
	KF.init(4, 2,   0);   //卡尔曼的4个动量参数和2个测量参数

	measurement = Mat_<float>::zeros(2, 1);
	measurement.at<float>(0, 0) = x;
	measurement.at<float>(0, 0) = y;

	KF.statePre.setTo(0);
	KF.statePre.at<float>(0, 0) = x;
	KF.statePre.at<float>(1, 0) = y;
	KF.statePost.setTo(0);   // 设置部分值为指定值
	KF.statePost.at<float>(0, 0) = x;
	KF.statePost.at<float>(1, 0) = y;

	setIdentity(KF.transitionMatrix);  //初始化一个缩放矩阵 s：分配给对角线的元素值
	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(.005)); //adjust this for faster convergence - but higher noise  快速收敛，噪音较大
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(.1));
}

Point kalmanPredict()
{
	Mat prediction = KF.predict();  //计算预测状态
	Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

	KF.statePre.copyTo(KF.statePost);
	KF.errorCovPre.copyTo(KF.errorCovPost);

	return predictPt;
}

Point kalmanCorrect(float x, float y)
{
	measurement(0) = x;
	measurement(1) = y;
	Mat estimated = KF.correct(measurement);  //更新预测状态
	Point statePt(estimated.at<float>(0), estimated.at<float>(1));  //更新后的值再传入
	return statePt;
}

int main()
{
	Mat frame, thresh_frame;
	int op = 0;
	vector<Mat> channels;
	
	// 设置视频及保存视频路径
	string save_path = "../video_save/test_result.mp4";
	string video_path = "../test.mp4";
	VideoCapture capture(video_path);

	// 添加，保存视频
	VideoWriter writer;
	Size size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
	int codec = VideoWriter::fourcc('m', 'p', '4', 'v');
	int rate = capture.get(cv::CAP_PROP_FPS);
	std::cout << "rate" << rate;	
	writer.open(save_path, codec, rate, size, true);

	int lo = 0;
	map<string,int> dict;
	// vector<int> vec1;
 
	if (!capture.isOpened())  // 成功打开视频 capture.isOpened() 返回true
		cerr << "Problem opening video source" << endl;  
	vector<Vec4i> hierarchy;  
	vector<vector<Point> > contours;   //vector 轮廓点储存再contours

	// 设置初始的背景，前景
	Mat back,img;
	Mat fore;
	int history=300;
	double varThreshold=16;
	bool detectShadows=false;

	Ptr<BackgroundSubtractorMOG2> bg = createBackgroundSubtractorMOG2(history, varThreshold, detectShadows);  //创建MOG背景减法器； history：用于训练背景的帧数，默认帧数为500帧； varThreshold:方差阈值，默认16.判断当前像素是前景还是背景

	int track = 0;
	bool update_bg_model = true;

	mousev.clear();
	kalmanv.clear();

	initKalman(0, 0);

	while (capture.grab())   // capture.grab() 获取视频帧
	{
		Point s, p;   // opencv 2维点模板s，p

		capture.retrieve(frame);
		bg->apply(frame, fore, -1);  //高斯混合，frame 原图（当前帧），fore：前景；update_bg_model为true(-1)（自动更新），为false：（0）不更新
		bg->getBackgroundImage(back);   // 背景图片结果
		erode(fore, fore, Mat());    //腐蚀源图像，取最小值的像素邻域的形状 
		dilate(fore, fore, Mat());  // 膨胀；

		normalize(fore, fore, 0, 1., NORM_MINMAX);
		threshold(fore, fore, .9, 1., THRESH_BINARY);  // 阈值 (阈值类型THRESH_BINARY，当大于设定值0.9，设为最大值，否则设为0)

		// 指针带返回值
		split(frame, channels);  // 分离通道
		add(channels[0], channels[1], channels[1]); 
		subtract(channels[2], channels[1], channels[2]);  //c2 = c2 - c1 -c0
		// threshold(channels[2], thresh_frame, 50, 255, CV_THRESH_BINARY);
		threshold(channels[2], thresh_frame, 50, 255, THRESH_BINARY);  // 将减完的c2通道根据阈值投影到（50，255）的像素间
		medianBlur(thresh_frame, thresh_frame, 7);  // 中值滤波；孔径线性尺寸，可以修改数值，来调整

		findContours(fore, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0)); // RETR_EXTERNAL：只检测最外围轮廓 ；RETR_LIST：检测所有轮廓（内，外）
		                                                                                    	 // CHAIN_APPROX_SIMPLE：保存物体边界上所有连续的轮廓点到contours向量内
		
																								 // Point(0, 0)：Point偏移量，所有的轮廓信息相对于原始图像对应点的偏移量，相当于在每一个检测出的轮廓点上加上该偏移量
																								 // contours 保存向量每组point的点集轮廓
		vector<vector<Point>> contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());

		//  Get the mass centers:  
		for (size_t i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true); // 将连续光滑的线条曲折化，即采用最小多边形包围物体（类似外接圆）； 第三个参数 epsilon越小，折线的形状越“接近”曲线
			boundRect[i] = boundingRect(Mat(contours_poly[i]));  // 用最小矩形将外围曲线包裹起来，返回最小矩形的(x,y,w,h)
		}

		p = kalmanPredict();
		// cout << "kalman prediction: " << p << "p.x" << p.x << "p.y " << p.y << endl;  
		mousev.push_back(p);
		string text;

		for (size_t i = 0; i < contours.size(); i++)
		{
			int point_value = 0;
			
			if (contourArea(contours[i]) > 500)       // 前景轮廓的面积大于1000就绘出相应的图形  // contours[i] 代表被检测物体的轮廓点
			{
				cout << "contourArea(contours[i]" << contourArea(contours[i]) << std::endl;
				rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0);   // 绘制物体的矩形框

				Point pt = Point(boundRect[i].x, boundRect[i].y+boundRect[i].height+15);
				Point pt1 = Point(boundRect[i].x, boundRect[i].y - 10);
				Point center = Point(boundRect[i].x + (boundRect[i].width / 2), boundRect[i].y + (boundRect[i].height / 2));
				circle(frame, center, 8, Scalar(0, 0, 255), -1, 1, 0);                               //绘制物体中心点
				circle(frame, Point(p.x, p.y), 8, Scalar(0, 255, 0), -1, 1, 0);      //绘制kalman 预测点
				Scalar color = CV_RGB(255, 0, 0);
				text = num2str(boundRect[i].width) + "*" + num2str(boundRect[i].height);
				putText(frame,"object", pt, cv::FONT_HERSHEY_DUPLEX, 1.0f, color);               // 绘制框下面 物体名称
				putText(frame,text, pt1, cv::FONT_HERSHEY_DUPLEX, 1.0f, color);                 // 绘制框上面 物体尺寸

				s = kalmanCorrect(center.x, center.y);   //s 卡尔曼更新后的中心坐标
				// drawCross(frame, s, Scalar(255, 255, 255), 5);

				int center_y1 = boundRect[i].y + (boundRect[i].height / 2);
				for (int i = mousev.size() - 50; i < mousev.size() - 1; i++)  //  mousev 经过卡尔曼预测的中心点变化的整个过程曲线,选取前50此kalman预测中心进行轨迹连线
				{
					line(frame, mousev[i], mousev[i + 1], Scalar(0, 255, 0), 1);
				}
			}
		}
		
		// 添加，保存视频
		writer << frame;

		// 视频展示
		// imshow("Video", frame);
	}
	return 0;

	
}
