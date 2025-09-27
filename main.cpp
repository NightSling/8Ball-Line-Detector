#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/fast_line_detector.hpp>
#include <vector>

using namespace cv;
using namespace std;

const int threshold_value = 230;
const double min_length = 20.0;    // minimum length to keep a line
const double extend_margin = 1000; // pixels to extend lines

inline double lineLength(const Vec4f &l) {
  return hypot(l[2] - l[0], l[3] - l[1]);
}

Vec4i extendLine(const Vec4f &l) {
  double dx = l[2] - l[0], dy = l[3] - l[1];
  double length = hypot(dx, dy);
  if (length == 0.0)
    return Vec4i(l[0], l[1], l[2], l[3]);

  double ux = dx / length, uy = dy / length;
  return Vec4i(int(l[0] - ux * extend_margin), int(l[1] - uy * extend_margin),
               int(l[2] + ux * extend_margin), int(l[3] + uy * extend_margin));
}

int main() {
  VideoCapture cap("/dev/video2", cv::CAP_V4L2);
  if (!cap.isOpened()) {
    cerr << "Cannot open camera" << endl;
    return -1;
  }

  Ptr<ximgproc::FastLineDetector> lsd = ximgproc::createFastLineDetector(
      15.0f, 1.414213562f, 50.0, 150.0, 7, true);

  Mat frame, gray, filtered;
  const double scaleFactor = 0.7;

  while (true) {
    if (!cap.read(frame) || frame.empty())
      break;

    Mat smallFrame;
    resize(frame, smallFrame, Size(), scaleFactor, scaleFactor, INTER_LINEAR);
    cvtColor(smallFrame, gray, COLOR_BGR2GRAY);

    // Threshold
    threshold(gray, filtered, threshold_value, 255, THRESH_BINARY);

    // Morphological operations
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(filtered, filtered, MORPH_CLOSE, kernel);
    dilate(filtered, filtered, kernel);

    // Edge enhancement
    Mat sharpened;
    Mat sharpKernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    filter2D(filtered, sharpened, -1, sharpKernel);
    filtered = sharpened;

    // Detect lines
    vector<Vec4f> lines;
    lsd->detect(filtered, lines);

    // Filter short segments and extend remaining
    vector<Vec4i> finalLines;
    for (const auto &l : lines) {
      if (lineLength(l) >= min_length)
        finalLines.push_back(extendLine(l));
    }

    // Draw lines directly on the original frame
    Mat output = smallFrame.clone();
    for (const auto &l : finalLines) {
      line(output, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2);
    }

    imshow("FLD Lines Highlighted", output);
    if ((waitKey(1) & 0xFF) == 'q')
      break;
  }

  cap.release();
  destroyAllWindows();
  return 0;
}
