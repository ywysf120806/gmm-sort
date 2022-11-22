#set(OPENCV_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rdparty)
set(OpenCV_LIBS)

LIST(APPEND OpenCV_LIBS ${PROJECT_SOURCE_DIR}/3rdparty/lib_4_6/libopencv_video.so.406)
LIST(APPEND OpenCV_LIBS ${PROJECT_SOURCE_DIR}/3rdparty/lib_4_6/libopencv_highgui.so.406)
LIST(APPEND OpenCV_LIBS ${PROJECT_SOURCE_DIR}/3rdparty/lib_4_6/libopencv_videoio.so.406)
LIST(APPEND OpenCV_LIBS ${PROJECT_SOURCE_DIR}/3rdparty/lib_4_6/libopencv_imgproc.so.406)
LIST(APPEND OpenCV_LIBS ${PROJECT_SOURCE_DIR}/3rdparty/lib_4_6/libopencv_core.so.406)