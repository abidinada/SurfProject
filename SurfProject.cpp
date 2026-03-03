#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <functional>
#include <numeric>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


struct Stats {
    string name;
    int kp1 = 0, kp2 = 0;
    int good = 0;
    int inliers = 0;
    double tDetectMs = 0.0;
    double tMatchMs = 0.0;
};


// Helper: Auto-scale display
void showScaled(const string& winName, const Mat& img, double maxWidth = 1200)
{
    double scale = 1.0;
    if (img.cols > maxWidth)
        scale = maxWidth / img.cols;

    Mat resized;
    resize(img, resized, Size(), scale, scale);

    namedWindow(winName, WINDOW_NORMAL);
    imshow(winName, resized);
}

// Fisheye Helpers

Mat createFisheyeMask(Size size)
{
    Mat mask = Mat::zeros(size, CV_8UC1);
    Point center(size.width / 2, size.height / 2);
    int radius = min(size.width, size.height) / 2;
    circle(mask, center, radius, Scalar(255), -1);
    return mask;
}

Mat enhanceCLAHE(const Mat& gray)
{
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    Mat out;
    clahe->apply(gray, out);
    return out;
}

void filterBorderKeypoints(vector<KeyPoint>& kps, Size size, float ratio = 0.9f)
{
    Point2f center(size.width / 2.f, size.height / 2.f);
    float maxRadius = (min(size.width, size.height) / 2.f) * ratio;

    vector<KeyPoint> filtered;
    for (const auto& kp : kps) {
        if (norm(kp.pt - center) < maxRadius)
            filtered.push_back(kp);
    }
    kps.swap(filtered);
}


// RANSAC Inliers

int countInliersHomography(const vector<KeyPoint>& kp1,
    const vector<KeyPoint>& kp2,
    const vector<DMatch>& matches)
{
    if (matches.size() < 4) return 0;

    vector<Point2f> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    Mat inlierMask;
    findHomography(pts1, pts2, RANSAC, 3.0, inlierMask);

    int inliers = 0;
    if (!inlierMask.empty())
        for (int i = 0; i < inlierMask.rows; i++)
            if (inlierMask.at<uchar>(i))
                inliers++;

    return inliers;
}

// SURF Classic

Stats runSURFClassic(const Mat& img1, const Mat& img2, Mat& visOut)
{
    Stats s;
    s.name = "SURF_classic";

    Mat g1, g2;
    cvtColor(img1, g1, COLOR_BGR2GRAY);
    cvtColor(img2, g2, COLOR_BGR2GRAY);

    Ptr<SURF> surf = SURF::create(2500, 4, 3, false, false);

    vector<KeyPoint> kp1, kp2;
    Mat d1, d2;

    TickMeter tm;
    tm.start();
    surf->detectAndCompute(g1, noArray(), kp1, d1);
    surf->detectAndCompute(g2, noArray(), kp2, d2);
    tm.stop();

    s.tDetectMs = tm.getTimeMilli();
    s.kp1 = kp1.size();
    s.kp2 = kp2.size();

    if (d1.empty() || d2.empty()) {
        visOut = img1.clone();
        return s;
    }

    tm.reset(); tm.start();
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knn;
    matcher.knnMatch(d1, d2, knn, 2);

    vector<DMatch> goodMatches;
    for (auto& m : knn)
        if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance)
            goodMatches.push_back(m[0]);

    tm.stop();

    s.tMatchMs = tm.getTimeMilli();
    s.good = goodMatches.size();
    s.inliers = countInliersHomography(kp1, kp2, goodMatches);

    drawMatches(img1, kp1, img2, kp2, goodMatches, visOut);
    return s;
}


// SURF Fisheye Adapted

Stats runSURFFisheyeAdapted(const Mat& img1, const Mat& img2, Mat& visOut)
{
    Stats s;
    s.name = "SURF_fisheye";

    Mat g1, g2;
    cvtColor(img1, g1, COLOR_BGR2GRAY);
    cvtColor(img2, g2, COLOR_BGR2GRAY);

    GaussianBlur(g1, g1, Size(3, 3), 0);
    GaussianBlur(g2, g2, Size(3, 3), 0);

    g1 = enhanceCLAHE(g1);
    g2 = enhanceCLAHE(g2);

    Mat mask1 = createFisheyeMask(g1.size());
    Mat mask2 = createFisheyeMask(g2.size());

    Ptr<SURF> surf = SURF::create(3500, 4, 3, false, false);

    vector<KeyPoint> kp1, kp2;
    Mat d1, d2;

    TickMeter tm;
    tm.start();
    surf->detectAndCompute(g1, mask1, kp1, d1);
    surf->detectAndCompute(g2, mask2, kp2, d2);
    tm.stop();

    filterBorderKeypoints(kp1, g1.size());
    filterBorderKeypoints(kp2, g2.size());

    surf->compute(g1, kp1, d1);
    surf->compute(g2, kp2, d2);

    s.tDetectMs = tm.getTimeMilli();
    s.kp1 = kp1.size();
    s.kp2 = kp2.size();

    if (d1.empty() || d2.empty()) {
        visOut = img1.clone();
        return s;
    }

    tm.reset(); tm.start();
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knn;
    matcher.knnMatch(d1, d2, knn, 2);

    vector<DMatch> goodMatches;
    for (auto& m : knn)
        if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance)
            goodMatches.push_back(m[0]);

    tm.stop();

    s.tMatchMs = tm.getTimeMilli();
    s.good = goodMatches.size();
    s.inliers = countInliersHomography(kp1, kp2, goodMatches);

    drawMatches(img1, kp1, img2, kp2, goodMatches, visOut);
    return s;
}


// ORB Classic

Stats runORBClassic(const Mat& img1, const Mat& img2, Mat& visOut)
{
    Stats s;
    s.name = "ORB_classic";

    Mat g1, g2;
    cvtColor(img1, g1, COLOR_BGR2GRAY);
    cvtColor(img2, g2, COLOR_BGR2GRAY);

    Ptr<ORB> orb = ORB::create(1000);

    vector<KeyPoint> kp1, kp2;
    Mat d1, d2;

    TickMeter tm;
    tm.start();
    orb->detectAndCompute(g1, noArray(), kp1, d1);
    orb->detectAndCompute(g2, noArray(), kp2, d2);
    tm.stop();

    s.tDetectMs = tm.getTimeMilli();
    s.kp1 = kp1.size();
    s.kp2 = kp2.size();

    if (d1.empty() || d2.empty()) {
        visOut = img1.clone();
        return s;
    }

    tm.reset(); tm.start();
    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> knn;
    matcher.knnMatch(d1, d2, knn, 2);

    vector<DMatch> goodMatches;
    for (auto& m : knn)
        if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance)
            goodMatches.push_back(m[0]);

    tm.stop();

    s.tMatchMs = tm.getTimeMilli();
    s.good = goodMatches.size();
    s.inliers = countInliersHomography(kp1, kp2, goodMatches);

    drawMatches(img1, kp1, img2, kp2, goodMatches, visOut);
    return s;
}


// Console Table

void printTable(const vector<Stats>& all)
{
    cout << "\n===== BENCHMARK RESULTS =====\n";
    cout << left << setw(18) << "Method"
        << setw(8) << "kp1"
        << setw(8) << "kp2"
        << setw(10) << "good"
        << setw(10) << "inliers"
        << setw(14) << "detect(ms)"
        << setw(14) << "match(ms)"
        << "\n";

    cout << string(82, '-') << "\n";

    for (const auto& s : all)
    {
        cout << left
            << setw(18) << s.name
            << setw(8) << s.kp1
            << setw(8) << s.kp2
            << setw(10) << s.good
            << setw(10) << s.inliers
            << setw(14) << fixed << setprecision(1) << s.tDetectMs
            << setw(14) << fixed << setprecision(1) << s.tMatchMs
            << "\n";
    }
}

// Generic Bar Chart

void drawFullGroupedBarChart(const vector<Stats>& results, const string& saveFolder)
{
    int width = 1700;
    int height = 850;

    Mat canvas(height, width, CV_8UC3, Scalar(255, 255, 255));

    int marginLeft = 120;
    int marginBottom = 170;
    int graphWidth = width - 300;
    int graphHeight = height - 300;

    Scalar axisColor(40, 40, 40);
    Scalar gridColor(220, 220, 220);

    Scalar surfClassicColor(200, 100, 50);
    Scalar surfFisheyeColor(50, 150, 250);
    Scalar orbColor(60, 180, 75);

    // Grid
    for (int i = 0;i <= 5;i++)
    {
        int y = height - marginBottom - (i * graphHeight / 5);
        line(canvas,
            Point(marginLeft, y),
            Point(marginLeft + graphWidth, y),
            gridColor, 1);
    }

    // Axes
    line(canvas,
        Point(marginLeft, height - marginBottom),
        Point(marginLeft + graphWidth, height - marginBottom),
        axisColor, 2);

    line(canvas,
        Point(marginLeft, height - marginBottom),
        Point(marginLeft, height - marginBottom - graphHeight),
        axisColor, 2);

    // Max value global
    double maxValue = 1.0;
    for (const auto& r : results)
    {
        maxValue = max(maxValue, (double)r.kp1);
        maxValue = max(maxValue, (double)r.kp2);
        maxValue = max(maxValue, (double)r.good);
        maxValue = max(maxValue, (double)r.inliers);
        maxValue = max(maxValue, r.tDetectMs);
        maxValue = max(maxValue, r.tMatchMs);
    }

    vector<string> groupNames =
    {
        "kp1", "kp2", "good", "inliers",
        "detect (ms)", "match (ms)"
    };

    int groupSpacing = graphWidth / (groupNames.size() + 1);
    int barWidth = 30;
    int baseY = height - marginBottom;

    for (int g = 0; g < groupNames.size(); g++)
    {
        int groupCenterX = marginLeft + groupSpacing * (g + 1);

        for (int i = 0; i < results.size(); i++)
        {
            double value = 0;

            if (g == 0) value = results[i].kp1;
            if (g == 1) value = results[i].kp2;
            if (g == 2) value = results[i].good;
            if (g == 3) value = results[i].inliers;
            if (g == 4) value = results[i].tDetectMs;
            if (g == 5) value = results[i].tMatchMs;

            int barHeight = (value / maxValue) * graphHeight;
            int x = groupCenterX - barWidth * 2 + i * (barWidth + 8);

            Scalar color = (i == 0) ? surfClassicColor :
                (i == 1) ? surfFisheyeColor :
                orbColor;

            rectangle(canvas,
                Point(x, baseY - barHeight),
                Point(x + barWidth, baseY),
                color, FILLED);

            putText(canvas,
                to_string((int)value),
                Point(x - 5, baseY - barHeight - 8),
                FONT_HERSHEY_SIMPLEX, 0.45,
                axisColor, 1);
        }

        putText(canvas,
            groupNames[g],
            Point(groupCenterX - 50, baseY + 50),
            FONT_HERSHEY_SIMPLEX, 0.6,
            axisColor, 2);
    }

    putText(canvas,
        "Complete Benchmark Comparison",
        Point(550, 90),
        FONT_HERSHEY_SIMPLEX, 1.1,
        axisColor, 3);

    imwrite(saveFolder + "/benchmark_chart.png", canvas);

    namedWindow("Full Benchmark", WINDOW_NORMAL);
    imshow("Full Benchmark", canvas);
}

int main()
{
    
    // Make execution deterministic
    // 
    cv::setNumThreads(1);
    cv::theRNG().state = 12345;

    Mat img1 = imread("Images/test2.bmp");
    Mat img2 = imread("Images/test3.bmp");

    if (img1.empty() || img2.empty()) {
        cout << "Error loading images!\n";
        return -1;
    }

    const int NUM_RUNS = 5;  // number of repetitions for averaging

    vector<Stats> surfClassicRuns;
    vector<Stats> surfFisheyeRuns;
    vector<Stats> orbRuns;

    Mat vis1, vis2, vis3;

   
    // Run multiple times
    
    for (int i = 0; i < NUM_RUNS; i++)
    {
        Mat tmp1, tmp2, tmp3;

        surfClassicRuns.push_back(runSURFClassic(img1, img2, tmp1));
        surfFisheyeRuns.push_back(runSURFFisheyeAdapted(img1, img2, tmp2));
        orbRuns.push_back(runORBClassic(img1, img2, tmp3));

        // Keep visualization from last run
        if (i == NUM_RUNS - 1)
        {
            vis1 = tmp1;
            vis2 = tmp2;
            vis3 = tmp3;
        }
    }

  
    // Compute averages
    
    auto averageStats = [](const vector<Stats>& runs)
        {
            Stats avg;
            avg.name = runs[0].name;

            for (const auto& r : runs)
            {
                avg.kp1 += r.kp1;
                avg.kp2 += r.kp2;
                avg.good += r.good;
                avg.inliers += r.inliers;
                avg.tDetectMs += r.tDetectMs;
                avg.tMatchMs += r.tMatchMs;
            }

            avg.kp1 /= runs.size();
            avg.kp2 /= runs.size();
            avg.good /= runs.size();
            avg.inliers /= runs.size();
            avg.tDetectMs /= runs.size();
            avg.tMatchMs /= runs.size();

            return avg;
        };

    vector<Stats> results;
    results.push_back(averageStats(surfClassicRuns));
    results.push_back(averageStats(surfFisheyeRuns));
    results.push_back(averageStats(orbRuns));

  
    // Print benchmark table
 
    printTable(results);

    // Create timestamp folder
    
    auto now = chrono::system_clock::now();
    auto timestamp = chrono::duration_cast<chrono::seconds>(
        now.time_since_epoch()).count();

    string runFolder = "Images/Results/" + to_string(timestamp);
    filesystem::create_directories(runFolder);

  
    // Save match images
    
    imwrite(runFolder + "/SURF_classic.png", vis1);
    imwrite(runFolder + "/SURF_fisheye.png", vis2);
    imwrite(runFolder + "/ORB_classic.png", vis3);

   
    // Save grouped benchmark chart
   
    drawFullGroupedBarChart(results, runFolder);

    cout << "\nAveraged over " << NUM_RUNS << " runs.\n";
    cout << "Results saved in folder: " << runFolder << endl;

    // -----------------------------
    // Display scaled images
    
    showScaled("SURF Classic", vis1);
    showScaled("SURF Fisheye", vis2);
    showScaled("ORB Classic", vis3);

    waitKey(0);
    destroyAllWindows();

    return 0;
}