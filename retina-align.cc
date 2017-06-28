/*
 * retina-align.cc
 *
 *  Created on: Jun 27, 2017
 *      Author: amyznikov
 */

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif


#include <stdio.h>
#include <stdarg.h>
#include <sys/stat.h>
#include "retina-align.h"
#include "debug.h"

using namespace cv;
using namespace std;


# define DEFAULT_MKDIR_MODE \
    (S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH)



#define CV_MTYPE_SWITCH(m, call, args) \
    switch ( m.type() ) { \
      case CV_8UC1  : return call<uint8_t> args; \
      case CV_8SC1  : return call<int8_t>  args; \
      case CV_16UC1 : return call<uint16_t> args; \
      case CV_16SC1 : return call<int16_t> args; \
      case CV_32FC1 : return call<float> args; \
      case CV_64FC1 : return call<double> args; \
      default : break;\
    }



///////////////////////////////////////////////////////////////////////////////

// REALTIME ms
static double __get_time(void)
{
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return (double)(t.tv_sec * 1000 + t.tv_nsec / 1e6);
}

// C-style string formating
static string __ssprintf(const char * format, ...)
{
  char * s = NULL;
  va_list arglist;
  string ss;

  va_start(arglist, format);
  if ( vasprintf(&s, format, arglist) > 0 ) {
    ss = s;
  }
  va_end(arglist);

  free(s);

  return ss;
}

/* get file name from full path name */
static string __extract_file_name(const string & fullpathname)
{
#ifdef _WIN32
  const char delims[] = "\\/";
#else
  const char delims[] = "/";
#endif
  size_t pos = fullpathname.find_last_of(delims);
  if ( pos != string::npos ) {
    return fullpathname.substr(pos + 1);
  }
  return fullpathname;
}

/* get file name part from full path name */
static std::string __extract_path(const std::string & fullpathname)
{
#ifdef _WIN32
  const char delims[] = "\\/";
#else
  const char delims[] = "/";
#endif
  size_t pos = fullpathname.find_last_of(delims);
  if ( pos != string::npos ) {
    return fullpathname.substr(0, pos);
  }
  return "";
}


static int __mkdir__(const char * path, mode_t mode)
{
#ifdef _WIN32
  return mkdir(path);
#else
  return mkdir(path, mode);
#endif
}

static bool __create_path(const std::string & path, mode_t mode = DEFAULT_MKDIR_MODE)
{
  size_t size;
  char tmp[(size = path.size()) + 1];

  if ( strcpy(tmp, path.c_str())[size - 1] == '/' ) {
    tmp[size - 1] = 0;
  }

  errno = 0;
  for ( char * p = tmp + 1; *p; p++ ) {
    if ( *p == '/' ) {
      *p = 0;
      if ( __mkdir__(tmp, mode) != 0 && errno != EEXIST ) {
        return false;
      }
      *p = '/';
    }
  }

  return __mkdir__(tmp, mode) == 0 || errno == EEXIST ? true : false;
}

static bool __save_image(InputArray _image, const string & fname)
{
  Mat converted;
  string dirname;

  if ( _image.getMat().empty() ) {
    CF_CRITICAL("empty image specified to save as '%s'", fname.c_str());
    return false;
  }

  if ( !(dirname = __extract_path(fname)).empty() ) {
    if ( !__create_path(dirname) ) {
      CF_CRITICAL("WARNING: create_path(%s) fails: %s", dirname.c_str(), strerror(errno));
    }
  }

  CF_DEBUG("C imwrite('%s')", fname.c_str());
  if (!imwrite(fname, _image)) {
    CF_FATAL("FATAL: imwrite(%s) fails", fname.c_str());
    return false;
  }

  return true;
}


///////////////////////////////////////////////////////////////////////////////

static inline void __imerode(Mat & image, int size, enum MorphShapes shape = MORPH_RECT)
{
  cv::erode(image, image, getStructuringElement(shape, Size(size, size)));
}

static void __mkCumulativeHist(Mat & gshist)
{
  double sum = 0;
  float * h;
  h = gshist.ptr<float>(0);
  for ( int i = 0; i < gshist.cols; ++i ) {
    sum += h[i];
    h[i] = (float) (sum);
  }
}

bool __calcHist(const Mat & image, Mat & gshist, int hbins, bool cummulative,
    bool scaled, double * minv, double * maxv, InputArray mask)
{
  minMaxLoc(image, minv, maxv, NULL, NULL, mask);
  if ( *minv >= *maxv || !isfinite(*minv) || !isfinite(*maxv) ) {
    CF_CRITICAL("invalid pixel ragne found : min=%g max=%g", *minv, *maxv);
    return false;
  }

  int histSize[] = { hbins };
  int channels[] = { 0 };
  float hrange[] = { (float) *minv, (float) *maxv + FLT_EPSILON };
  const float * ranges[] = { hrange };

  cv::calcHist(&image, 1, channels, mask, gshist, 1, histSize, ranges, true, false);
  if ( gshist.empty() ) {
    CF_CRITICAL("cv::calcHist() fails");
    return false;
  }

  if ( gshist.cols == 1 ) {
    transpose(gshist, gshist);
  }

  if ( scaled ) {
    divide(gshist, cv::sum(gshist), gshist);
  }

  if ( cummulative ) {
    __mkCumulativeHist(gshist);
  }

  return true;
}


static inline double __sat(double v, double min, double max)
{
  return v < min ? min : v > max ? max : v;
}

template<class T>
static void __normalize_(Mat & image, double imin, double imax, double omin, double omax, InputArray _mask = noArray())
{
  T * row;
  if ( imin >= imax ) {
    minMaxLoc(image, &imin, &imax, 0, 0, _mask);
  }
  const double scale = (omax - omin) / (imax - imin);
  for ( int y = 0; y < image.rows; y++ ) {
    row = (T *) (image.data + image.step.p[0] * y);
    for ( int x = 0; x < image.cols; x++ ) {
      row[x] = (T) __sat((row[x] - imin) * scale + omin, omin, omax);
    }
  }
}

static void __normalize(Mat & image, double imin, double imax, double omin, double omax, InputArray _mask = noArray())
{
  CV_MTYPE_SWITCH(image, __normalize_, (image, imin, imax, omin, omax, _mask));
}



static bool __stretch_histogram(Mat & image, double plo, double phi, double omin, double omax, InputArray mask = noArray())
{
  Mat gshist;
  int nbins = 0, lowidx, highidx;
  double minv = 0, maxv = 0, range, hmin, imin, imax;

  nbins = 100;

  plo /= 100;
  phi /= 100;

  if ( !__calcHist(image, gshist, nbins, true, true, &minv, &maxv, mask) ) {
    CF_CRITICAL("calcGrayscaleHist() fails");
    return false;
  }

  if ( (range = (maxv - minv)) <= 0 ) {
    CF_CRITICAL("__calcHist() returns empty pixel value range");
    return false;
  }

  hmin = gshist.at<float>(0, lowidx = 0);
  while ( lowidx < nbins && gshist.at<float>(0, lowidx) == hmin ) {
    ++lowidx;
  }
  hmin = gshist.at<float>(0, lowidx);
  while ( lowidx < nbins && gshist.at<float>(0, lowidx) - hmin < plo ) {
    ++lowidx;
  }
  imin = minv + lowidx * range / nbins;


  highidx = lowidx + 1;
  while ( highidx < nbins && gshist.at<float>(0, highidx) < phi ) {
    ++highidx;
  }
  imax = minv + highidx * range / nbins;

  if ( omax > omin ) {
   __normalize(image, imin, imax, omin, omax, mask);
  }
  else {
    image.setTo(imin, image < imin);
    image.setTo(imax, image > imax);
  }

  return true;
}


static void __frangi2D(InputArray _src, Mat & _dst, double sigma, Mat * _mask, bool equalize = true)
{
  Mat src, dst;
  Mat Dxx, Dyy, Dxy;
  Mat Tr, T;
  const int ksize = 3;

  if ( sigma > 0 ) {
    GaussianBlur(_src, src, Size(0, 0), sigma);
  }
  else {
    src = _src.getMat();
  }

  Sobel(src, Dxx, CV_32FC1, 2, 0, ksize);
  Sobel(src, Dyy, CV_32FC1, 0, 2, ksize);
  Sobel(src, Dxy, CV_32FC1, 1, 1, ksize);

  Tr = Dxx + Dyy;
  T = Dxx - Dyy;
  sqrt(T.mul(T) + 4.0 * Dxy.mul(Dxy), T); // T = sqrt( (Dxx - Dyy)^2 + 4 * Dxy^2)
  cv::max((Tr - T), 0.0, dst);
  cv::max((Tr + T), dst, dst);

  if ( _mask && !_mask->empty() ) {
    __imerode(*_mask, ((int) (sigma * 3)) | 1);
    dst.setTo(0, ~*_mask);
  }

  if ( equalize ) {
    const int w = std::max((dst.cols / 5) | 1, (dst.rows / 5) | 1);
    if ( w > 150 ) {
      blur(dst, T, Size(w, w));
      /* workaround potential division by zero on large saturated areas */
      add(T, cv::mean(dst).val[0] * 0.5 + 10 * FLT_EPSILON, T);
      divide(dst, T, dst);
    }
  }

  _dst = dst;
}


static Ptr<Feature2D> __create_keypoints_detector(void)
{
  int nfeatures = 500;
  float scaleFactor = 1.2f;
  int nlevels = 8;
  int edgeThreshold = 1;
  int firstLevel = 0;
  int WTA_K = 2;
  int scoreType = ORB::HARRIS_SCORE;
  int patchSize = 31;
  int fastThreshold = 20;

  return ORB::create(nfeatures = 5000,
      scaleFactor = 1.25f,
      nlevels = 4,
      edgeThreshold = 5,
      firstLevel = 0,
      WTA_K = 4,
      scoreType = ORB::HARRIS_SCORE,
      patchSize = 75,
      fastThreshold = 20
      );
}


static size_t __extract_keypoints(Ptr<Feature2D> detector, const Mat & image, const Mat & _mask,
    Mat * outImg, Mat * outMask, vector<KeyPoint> * keypoints, Mat * descriptors )
{
  Mat src, mask;

  if ( !_mask.empty() ) {
    mask = _mask.clone();
  }
  else {
    mask = Mat(image.size(), CV_8UC1, 255);
  }

  __frangi2D(image, src, 3, &mask, true);
  __stretch_histogram(src, 1, 99, 0, 255, _mask);

  if ( outImg ) {
    *outImg = src.clone();
  }

  if ( outMask ) {
    *outMask = mask.clone();
  }

  src.convertTo(src, CV_8UC1);

  keypoints->clear();
  descriptors->release();

  if ( !mask.empty() ) {
    __imerode(mask, 35); // move away from edges
  }

  detector->detectAndCompute(src, mask, *keypoints, *descriptors, false);

  return keypoints->size();
}


static inline double __dang(double a1, double a2)
{
  double da = fabs(a2 - a1);
  return da > 180 ? 360 - da : da;
}

static size_t __match_key_points(DescriptorMatcher & matcher, const Mat & descriptors_1, const Mat & descriptors_2, vector<DMatch> * matches)
{
  matcher.match(descriptors_1, descriptors_2, *matches);
  if ( matches->size() > 0 ) {
    sort(matches->begin(), matches->end(), less<DMatch>());
  }
  return matches->size();
}

static size_t __select_best_matches(const vector<KeyPoint> & keypoints_1, const vector<KeyPoint> & keypoints_2,
    const vector<DMatch> & all_matches, vector<DMatch> * good_matches,
    vector<Point2f> * points1, vector<Point2f> * points2,
    size_t nmax)
{
  size_t nb_matches = 0;

  if ( good_matches ) {
    good_matches->clear();
  }

  if ( points1 ) {
    points1->clear();
  }

  if ( points2 ) {
    points2->clear();
  }

  if ( all_matches.size() > 0 ) {

    const double mindist = all_matches[0].distance;

    for ( size_t i = 0, n = all_matches.size(); i < n && nb_matches < nmax; ++i ) {

      const DMatch & m = all_matches[i];

      if ( m.distance > 15 * mindist && nb_matches > 7 ) {
        break;
      }

      const KeyPoint & kp1 = keypoints_1[m.queryIdx];
      const KeyPoint & kp2 = keypoints_2[m.trainIdx];

      if ( __dang(kp1.angle, kp2.angle) > 60 ) {
        continue;
      }

      if ( points1 ) {
        points1->emplace_back(kp1.pt);
      }

      if ( points2 ) {
        points2->emplace_back(kp2.pt);
      }

      if ( good_matches ) {
        good_matches->emplace_back(m);
      }

      ++nb_matches;
    }
  }

  return nb_matches;
}

template<class T>
static inline Point2f __affine_tansform_(const Mat & M, const Point2f & in) {
  const T * m = M.ptr<T>(0);
  return Point2f(m[0] * in.x + m[1] * in.y + m[2], m[3] * in.x + m[4] * in.y + m[5]);
}


template<class T>
static inline void __affine_tansform_(const Mat & M, const vector<Point2f> & vin, vector<Point2f> * vout)
{
  vout->resize(vin.size());
  for ( size_t i = 0; i < vin.size(); ++i ) {
    (*vout)[i] = __affine_tansform_<T>(M, vin[i]);
  }
}

static inline Point2f __affine_tansform(const Mat & M, const Point2f & in)
{
  switch ( M.type() ) {
    case CV_32FC1 :
    return __affine_tansform_<float>(M, in);
    case CV_64FC1 :
    return __affine_tansform_<double>(M, in);
    default :
    break;
  }
  return Point2f();
}

static inline void __affine_tansform(const Mat & M, const vector<Point2f> & vin, vector<Point2f> * vout)
{
  switch ( M.type() ) {
    case CV_32FC1 :
    __affine_tansform_<float>(M, vin, vout);
    break;
    case CV_64FC1 :
    __affine_tansform_<double>(M, vin, vout);
    break;
    default :
    break;
  }
}

static bool __getAffineMatrix(const Point2f a[], const Point2f b[], int count, Mat & M)
{
  float sa[6][6] = { { 0. } }, sb[6] = { 0. };
  Mat A(6, 6, CV_32F, &sa[0][0]), B(6, 1, CV_32F, sb);
  Mat MM = M.reshape(1, 6);

  for ( int i = 0; i < count; i++ ) {
    sa[0][0] += a[i].x * a[i].x;
    sa[0][1] += a[i].y * a[i].x;
    sa[0][2] += a[i].x;

    sa[1][1] += a[i].y * a[i].y;
    sa[1][2] += a[i].y;

    sb[0] += a[i].x * b[i].x;
    sb[1] += a[i].y * b[i].x;
    sb[2] += b[i].x;
    sb[3] += a[i].x * b[i].y;
    sb[4] += a[i].y * b[i].y;
    sb[5] += b[i].y;
  }

  sa[3][4] = sa[4][3] = sa[1][0] = sa[0][1];
  sa[3][5] = sa[5][3] = sa[2][0] = sa[0][2];
  sa[4][5] = sa[5][4] = sa[2][1] = sa[1][2];

  sa[3][3] = sa[0][0];
  sa[4][4] = sa[1][1];
  sa[5][5] = sa[2][2] = count;

  try {
    cv::solve(A, B, MM, DECOMP_EIG);
  }
  catch ( const Exception & e ) {
    CF_FATAL("cv::solve() fails: %s", e.msg.c_str());
    M.release();
    return false;
  }

  return true;

}

static bool __getX2Y2Matrix(const Point2f a[], const Point2f b[], int N, Mat & M)
{
  // x' =  a00 * x + a01 * y + a02 + a03 * x * y + a04 * x*x + a05 * y*y
  // y' =  a10 * x + a11 * y + a12 + a13 * x * y + a14 * x*x + a15 * y*y
  float * s;

  Mat src1(N, 6, CV_32FC1);
  Mat src2_x(N, 1, CV_32FC1);
  Mat src2_y(N, 1, CV_32FC1);

  M.release();

  for ( int i = 0; i < N; ++i ) {
    s = src1.ptr<float>(i);
    s[0] = a[i].x;
    s[1] = a[i].y;
    s[2] = 1;
    s[3] = a[i].x * a[i].y;
    s[4] = a[i].x * a[i].x;
    s[5] = a[i].y * a[i].y;
  }
  for ( int i = 0; i < N; ++i ) {
    s = src2_x.ptr<float>(i);
    s[0] = b[i].x;
  }
  for ( int i = 0; i < N; ++i ) {
    s = src2_y.ptr<float>(i);
    s[0] = b[i].y;
  }

  /* src2 = src1 * M */

  try {

    Mat Mx, My;

    cv::solve(src1, src2_x, Mx, DECOMP_NORMAL);
    cv::solve(src1, src2_y, My, DECOMP_NORMAL);

    Mx = Mx.t();
    My = My.t();

    M = Mat (2, 6, CV_32FC1);
    memcpy(M.ptr<float>(0), Mx.ptr<float>(0), 6 * sizeof(float));
    memcpy(M.ptr<float>(1), My.ptr<float>(0), 6 * sizeof(float));
  }
  catch (const Exception & e) {
    CF_FATAL("cv::solve() fails: %s", e.msg.c_str());
    M.release();
    return false;
  }

  return true;
}

// Based on RANSAC estimateRigidTransform()
static Mat __estimate_affine_transform(vector<Point2f> & pA, vector<Point2f> & pB, double minsep, double good_eps,
    const vector<DMatch> * matches = NULL, vector<DMatch> * best_matches = NULL)
{
  Mat M(2, 3, CV_32FC1);

  const int RANSAC_MAX_ITERS = 1500;
  const int RANSAC_SIZE0 = 6;
  //const double RANSAC_GOOD_RATIO = 0.35;
  const double RANSAC_EXCELLENT_RATIO = 0.75;
  double S1;
  double S2;

  int good_count = 0;
  std::vector<int> good_idx;

  int best_count = 0;
  std::vector<int> best_idx;

  int i, j, k, k1;

  RNG rng((uint64)-1); //time(0)

  int count = pA.size();

  good_idx.resize(count);

  if ( count < RANSAC_SIZE0 ) {
    return Mat();
  }

  // RANSAC stuff:
  // 1. find the consensus
  for ( k = 0; k < RANSAC_MAX_ITERS; k++ ) {

    int idx[RANSAC_SIZE0];
    Point2f a[RANSAC_SIZE0];
    Point2f b[RANSAC_SIZE0];

    // choose random 3 non-complanar points from A & B
    for ( i = 0; i < RANSAC_SIZE0; i++ ) {

      for ( k1 = 0; k1 < RANSAC_MAX_ITERS; k1++ ) {

        idx[i] = rng.uniform(0, count);

        for ( j = 0; j < i; j++ ) {

          if ( idx[j] == idx[i] ) {
            break;
          }

          // check that the points are not very close one each other
          if ( hypot(pA[idx[i]].x - pA[idx[j]].x, pA[idx[i]].y - pA[idx[j]].y) < minsep ) {
            break;
          }
          if ( hypot(pB[idx[i]].x - pB[idx[j]].x, pB[idx[i]].y - pB[idx[j]].y) < minsep ) {
            break;
          }
        }

        if ( j < i ) {
          continue;
        }



        if ( i + 1 == RANSAC_SIZE0 ) {
          // additional check for non-complanar vectors
          a[0] = pA[idx[0]];
          a[1] = pA[idx[1]];
          a[2] = pA[idx[2]];

          b[0] = pB[idx[0]];
          b[1] = pB[idx[1]];
          b[2] = pB[idx[2]];

          double dax1 = a[1].x - a[0].x, day1 = a[1].y - a[0].y;
          double dax2 = a[2].x - a[0].x, day2 = a[2].y - a[0].y;
          double dbx1 = b[1].x - b[0].x, dby1 = b[1].y - b[0].y;
          double dbx2 = b[2].x - b[0].x, dby2 = b[2].y - b[0].y;
          const double eps = 0.01;

          if ( fabs(dax1 * day2 - day1 * dax2) < eps * hypot(dax1, day1) * hypot(dax2, day2)
              || fabs(dbx1 * dby2 - dby1 * dbx2) < eps * hypot(dbx1, dby1) * hypot(dbx2, dby2) ) {
            continue;
          }

        }
        break;
      }

      if( k1 >= RANSAC_MAX_ITERS ) {
        break;
      }
    }

    if ( i < RANSAC_SIZE0 ) {
      continue;
    }

    // estimate the transformation using 3 points
    __getAffineMatrix(a, b, 3, M);

    if ( M.at<float>(0, 0) < 0 || M.at<float>(1, 1) < 0 || M.at<float>(0, 1) * M.at<float>(1, 0) > 0 ) {
      continue;
    }

    S1 = M.at<float>(0, 0);
    S2 = M.at<float>(1, 1);
    if ( S1 < 0.4 * S2 || S2 < 0.4 * S1 ) {
      continue;
    }

    S1 = hypot(M.at<float>(0, 0), M.at<float>(1, 0));
    S2 = hypot(M.at<float>(0, 1), M.at<float>(1, 1));
    if ( S1 < 0.4 || S1 > 1/0.4 || S2 < 0.4 || S2 > 1/0.4 || S1 < 0.4 * S2 || S2 < 0.4 * S1 ) {
      continue;
    }

    for ( i = 0, good_count = 0; i < count; i++ ) {
      const Point2f p = __affine_tansform(M, pA[i]);
      if ( hypot(p.x - pB[i].x, p.y - pB[i].y) < good_eps ) {
        good_idx[good_count++] = i;
      }
    }

    if ( good_count > best_count ) {
      best_count = good_count;
      best_idx = good_idx;
    }

    if ( best_count >= count * RANSAC_EXCELLENT_RATIO ) {
      break;
    }
  }


  if ( k >= RANSAC_MAX_ITERS && best_count < 4 /*(int) std::max(5.0, count * RANSAC_GOOD_RATIO)*/) {
    return Mat();
  }

  if ( best_count < count ) {
    for ( i = 0; i < best_count; i++ ) {
      j = best_idx[i];
      pA[i] = pA[j];
      pB[i] = pB[j];
      if ( matches && best_matches ) {
        best_matches->emplace_back((*matches)[j]);
      }
    }
    pA.erase(pA.begin() + best_count, pA.end());
    pB.erase(pB.begin() + best_count, pB.end());
  }
  else if ( matches && best_matches ) {
    *best_matches = *matches;
  }

  __getAffineMatrix(&pA[0], &pB[0], good_count, M);

  return M;
}


static Mat __estimate_transform(const vector<KeyPoint> & master_keypoints, const vector<KeyPoint> & image_keypoints,
    const vector<DMatch> & all_matches, vector<DMatch> * good_matches)
{
  vector<Point2f> master_points, image_points;
  size_t nmax;

  const double GOOD_EPS = 5;

  nmax = min((size_t) 15, all_matches.size());
  while ( nmax <= all_matches.size() ) {

    vector<DMatch> best_matches;
    Mat M;

    __select_best_matches(master_keypoints, image_keypoints, all_matches, good_matches, &master_points, &image_points, nmax);

    if ( good_matches->size() > 2 ) {
      if ( !(M = __estimate_affine_transform(master_points, image_points, 7, GOOD_EPS, good_matches, &best_matches)).empty() ) {

        /* gather total good matches */
        master_points.clear();
        image_points.clear();
        good_matches->clear();

        double xmin = 0, xmax = 0, ymin = 0, ymax = 0;

        for ( size_t i = 0; i < all_matches.size(); ++i ) {
          const DMatch & m = all_matches[i];
          const KeyPoint & kp1 = master_keypoints[m.queryIdx];
          const KeyPoint & kp2 = image_keypoints[m.trainIdx];
          Point2f p1 = kp1.pt;
          Point2f p2 = kp2.pt;
          Point2f pt = __affine_tansform(M, p1);
          if ( hypot(pt.x - p2.x, pt.y - p2.y) < 5 ) {

            if ( !good_matches->size() ) {
              xmin = p1.x;
              xmax = p1.x;
              ymin = p1.y;
              ymax = p1.y;
            }
            else {
              if ( p1.x < xmin ) {
                xmin = p1.x;
              }
              else if ( p1.x > xmax ) {
                xmax = p1.x;
              }
              if ( p1.y < ymin ) {
                ymin = p1.y;
              }
              else if ( p1.y > ymax ) {
                ymax = p1.y;
              }
            }

            master_points.emplace_back(p1);
            image_points.emplace_back(p2);
            good_matches->emplace_back(m);
          }
        }

        if ( good_matches->size() > 2 ) {
          if ( xmax - xmin > 500 && ymax - ymin > 500 && good_matches->size() > 50 ) {
            __getX2Y2Matrix(&*master_points.begin(), &*image_points.begin(), image_points.size(), M);
          }
          else {
            __getAffineMatrix(&*master_points.begin(), &*image_points.begin(), image_points.size(), M);
          }

          if ( !M.empty() ) {
            if ( M.cols != 6 ) {
              M = affine2x2y2(M);
            }
            return M;
          }
        }
      }
    }

    if ( nmax > 30 ) {
      nmax = max(all_matches.size(), (size_t) (1.5 * nmax));
    }
    else {
      nmax += 5;
    }
  }

  return Mat();
}



///////////////////////////////////////////////////////////////////////////////

static void __jacobian_x2y2(const Mat& gradientXWarped, const Mat& gradientYWarped,
    const struct X2Y2Grid & g, Mat & jac)
{
  const int w = gradientXWarped.cols;

  jac.colRange(0, w) = gradientXWarped.mul(g.Xgrid);    //1
  jac.colRange(w, 2 * w) = gradientYWarped.mul(g.Xgrid);    //2
  jac.colRange(2 * w, 3 * w) = gradientXWarped.mul(g.Ygrid);    //3
  jac.colRange(3 * w, 4 * w) = gradientYWarped.mul(g.Ygrid);    //4
  gradientXWarped.copyTo(jac.colRange(4 * w, 5 * w));    //5
  gradientYWarped.copyTo(jac.colRange(5 * w, 6 * w));    //6
  jac.colRange(6 * w, 7 * w) = gradientXWarped.mul(g.XYgrid);    //7
  jac.colRange(7 * w, 8 * w) = gradientYWarped.mul(g.XYgrid);    //8
  jac.colRange(8 * w, 9 * w) = gradientXWarped.mul(g.X2grid);    //9
  jac.colRange(9 * w, 10 * w) = gradientYWarped.mul(g.X2grid);    //10
  jac.colRange(10 * w, 11 * w) = gradientXWarped.mul(g.Y2grid);    //11
  jac.colRange(11 * w, 12 * w) = gradientYWarped.mul(g.Y2grid);    //10
}


/* this functions is used for two types of projections. If src1.cols ==src.cols
 it does a blockwise multiplication (like in the outer product of vectors)
 of the blocks in matrices src1 and src2 and dst
 has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
 (number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.

 The number_of_blocks is equal to the number of parameters we are lloking for
 (i.e. rtanslation:2, euclidean: 3, affine: 6, homography: 8)
 */
static void __project_onto_jacobian(const Mat& src1, const Mat& src2, Mat& dst)
{
  int w;
  double nn;

  float * dstPtr = dst.ptr<float>(0);

  if (src1.cols != src2.cols) {// dst.cols==1
    w = src2.cols;
    for (int i = 0; i < dst.rows; i++) {
      dstPtr[i] = (float) src2.dot(src1.colRange(i * w, (i + 1) * w));
    }
  }
  else {
    w = src2.cols / dst.cols;
    for (int i = 0; i < dst.rows; i++) {
      Mat mat(src1.colRange(i * w, (i + 1) * w));
      nn = norm(mat);
      dstPtr[i * (dst.rows + 1)] = nn * nn; //diagonal elements
      for (int j = i + 1; j < dst.cols; j++) { //j starts from i+1
        dstPtr[i * dst.cols + j] = mat.dot(src2.colRange(j * w, (j + 1) * w));
        dstPtr[j * dst.cols + i] = dstPtr[i * dst.cols + j]; //due to symmetry
      }
    }
  }
}

static void __update_warp_matrix_x2y2(Mat & map_matrix, const Mat & update)
{
  float * mapPtr = map_matrix.ptr<float>(0);
  const float* updatePtr = update.ptr<float>(0);
  mapPtr[0] += updatePtr[0];
  mapPtr[6] += updatePtr[1];
  mapPtr[1] += updatePtr[2];
  mapPtr[7] += updatePtr[3];
  mapPtr[2] += updatePtr[4];
  mapPtr[8] += updatePtr[5];
  mapPtr[3] += updatePtr[6];
  mapPtr[9] += updatePtr[7];
  mapPtr[4] += updatePtr[8];
  mapPtr[10] += updatePtr[9];
  mapPtr[5] += updatePtr[10];
  mapPtr[11] += updatePtr[11];
}


static bool __ecc_align(const Mat & image1, const Mat & mask1, const Mat & image2, const Mat & mask2, Mat & warp,
    int maxIters, double eps, int * numiterations, double * rho )
{
  const Mat & src = image1; //template image
  const Mat & dst = image2; //input image (to be warped)
  Mat src_mask = mask1.clone(); //template mask
  Mat dst_mask = mask2.clone(); //input mask (to be warped)

  const int numberOfParameters = 12;

  const int ws = src.cols;
  const int hs = src.rows;
  const int wd = dst.cols;
  const int hd = dst.rows;

  struct X2Y2Grid g;

  if ( eps <= 0 ) {
    eps = 1e-5;
  }

  if ( warp.empty() ) {
    warp = Mat::eye(2, 6, CV_32FC1);
  }

  if ( warp.type() == CV_64FC1 ) {
    warp.convertTo(warp, CV_32FC1);
  }

  if (warp.type() != CV_32FC1) {
    CF_FATAL("Warp Matrix must be single-channel floating-point matrix");
    return false;
  }

  if (warp.rows != 2 || warp.cols != 6) {
    CF_FATAL("Input matrix must have 2x6 size for motion X2Y2");
    return false;
  }

  create_x2y2_grid(&g, src.size());

  Mat templateZM = Mat(hs, ws, CV_32F); // to store the (smoothed)zero-mean version of template
  Mat templateFloat = Mat(hs, ws, CV_32F); // to store the (smoothed) template
  Mat imageFloat = Mat(hd, wd, CV_32F); // to store the (smoothed) input image
  Mat imageWarped = Mat(hs, ws, CV_32F); // to store the warped zero-mean input image
  Mat imageMask = Mat(hs, ws, CV_8U); //to store the final mask


  src.convertTo(templateFloat, templateFloat.type());
  GaussianBlur(templateFloat, templateFloat, Size(0, 0), 1);

  dst.convertTo(imageFloat, imageFloat.type());
  GaussianBlur(imageFloat, imageFloat, Size(0, 0), 1);


  // to use it for mask warping
  if (dst_mask.empty()) {
    //dst_mask = Mat::ones(hd, wd, CV_8U);
    dst_mask = Mat(hd, wd, CV_8UC1, 255);
  }
  else {
    __imerode(dst_mask, 5);
  }

  if ( !src_mask.empty() ) {
    __imerode(src_mask, 5);
  }


  // needed matrices for gradients and warped gradients
  Mat gradientX = Mat::zeros(hd, wd, CV_32FC1);
  Mat gradientY = Mat::zeros(hd, wd, CV_32FC1);
  Mat gradientXWarped = Mat(hs, ws, CV_32FC1);
  Mat gradientYWarped = Mat(hs, ws, CV_32FC1);

  // calculate first order image derivatives
  Matx13f dx(-0.5f, 0.0f, 0.5f);

  filter2D(imageFloat, gradientX, -1, dx);
  filter2D(imageFloat, gradientY, -1, dx.t());

  gradientX.setTo(0, ~dst_mask);
  gradientY.setTo(0, ~dst_mask);


  // matrices needed for solving linear equation system for maximizing ECC
  Mat jacobian = Mat(hs, ws * numberOfParameters, CV_32F);
  Mat hessian = Mat(numberOfParameters, numberOfParameters, CV_32F);
  Mat hessianInv = Mat(numberOfParameters, numberOfParameters, CV_32F);
  Mat imageProjection = Mat(numberOfParameters, 1, CV_32F);
  Mat templateProjection = Mat(numberOfParameters, 1, CV_32F);
  Mat imageProjectionHessian = Mat(numberOfParameters, 1, CV_32F);
  Mat errorProjection = Mat(numberOfParameters, 1, CV_32F);

  Mat deltaP = Mat(numberOfParameters, 1, CV_32F); //transformation parameter correction
  Mat error = Mat(hs, ws, CV_32F); //error as 2D matrix

  const int imageFlags = INTER_LINEAR;
  const int maskFlags = INTER_NEAREST;

  // iteratively update map_matrix
  *rho = -1;
  double last_rho = -eps;

  Mat map_x, map_y;
  Scalar imgMean, imgStd, tmpMean, tmpStd;

//  double t1, t2;

  for ( * numiterations = 0; * numiterations < maxIters && fabs(*rho - last_rho) >= eps; ++ *numiterations ) {

    Mat map_x, map_y;

    int borderMode = BORDER_CONSTANT;
    const Scalar borderValue = Scalar(0);

    //
//    t1 = get_time();
    if ( !create_x2y2_remap(g, warp, map_x, map_y) ) {
      CF_FATAL("__create_x2y2_remap() fails.");
      return false;
    }

    remap(imageFloat, imageWarped, map_x, map_y, imageFlags, borderMode, borderValue);
    remap(gradientX, gradientXWarped, map_x, map_y, imageFlags, borderMode, borderValue);
    remap(gradientY, gradientYWarped, map_x, map_y, imageFlags, borderMode, borderValue);
    remap(dst_mask, imageMask, map_x, map_y, maskFlags, borderMode, borderValue);

//    t2 = get_time();
//    CF_DEBUG("REMAP: %g ms", t2 - t1);
    //


    //
//    t1 = get_time();

    if ( !src_mask.empty() ) {
      imageMask &= src_mask;
    }

    gradientXWarped.setTo(0, ~imageMask);
    gradientYWarped.setTo(0, ~imageMask);

    meanStdDev(imageWarped, imgMean, imgStd, imageMask);
    subtract(imageWarped, imgMean, imageWarped, imageMask); //zero-mean input
    imageWarped.setTo(0, ~imageMask);

    meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);
    //templateZM.setTo(Scalar(0.0f));
    subtract(templateFloat, tmpMean, templateZM, imageMask); //zero-mean template
    templateZM.setTo(0, ~imageMask);

    const int cnz = countNonZero(imageMask);
    const double imgNorm = std::sqrt( cnz * (imgStd.val[0]) * (imgStd.val[0]));
    const double tmpNorm = std::sqrt( cnz * (tmpStd.val[0]) * (tmpStd.val[0]));

//    t2 = get_time();
//    CF_DEBUG("NORMS: %g ms", t2 - t1);
    //

    //
//    t1 = get_time();

    // calculate jacobian of image wrt parameters
    __jacobian_x2y2(gradientXWarped, gradientYWarped, g, jacobian);

//    t2 = get_time();
//    CF_DEBUG("JACOBIAN: %g ms", t2 - t1);
    //



    //
//    t1 = get_time();

    const double correlation = templateZM.dot(imageWarped);
//    save_image(templateZM, ssprintf("templateZM/templateZM-%d.png", *numiterations), CV_8UC1, 0, 255);
//    save_image(imageWarped, ssprintf("imageWarped/imageWarped-%d.png", *numiterations), CV_8UC1, 0, 255);


    // calculate enhanced correlation coefficient (ECC)->rho
    last_rho = *rho;
    *rho = correlation / (imgNorm * tmpNorm);
    if ( cvIsNaN(*rho) ) {
      CF_FATAL("NaN encountered. correlation=%g imgNorm=%g tmpNorm=%g tmpStd=%g", correlation, imgNorm, tmpNorm, tmpStd.val[0]);
      return false;
    }


    // project images into jacobian
    __project_onto_jacobian(jacobian, imageWarped, imageProjection);


    __project_onto_jacobian(jacobian, templateZM, templateProjection);

    //t1 = get_time();
    // calculate Hessian and its inverse
    __project_onto_jacobian(jacobian, jacobian, hessian);
    //t2 = get_time();
    //CF_DEBUG("__project_onto_jacobian: %g ms", t2 - t1);

    hessianInv = hessian.inv();


    // calculate the parameter lambda to account for illumination variation
    imageProjectionHessian = hessianInv * imageProjection;

    const double dot1 = imageProjection.dot(imageProjectionHessian);
    const double lambda_n = (imgNorm * imgNorm) - dot1;

    const double dot2 = templateProjection.dot(imageProjectionHessian);
    const double lambda_d = correlation - dot2;

    if (lambda_d <= 0.0) {

      *rho = -1;

      CF_FATAL("The algorithm stopped before its convergence.\n"
          "The correlation is going to be minimized.\n"
          "Images may be uncorrelated or non-overlapped.\n"
          "correlation=%g lambda_n=%g lambda_d=%g iteration=%d imgNorm=%g tmpNorm=%g dot1=%g dot2=%g",
          correlation, lambda_n, lambda_d, * numiterations, imgNorm, tmpNorm, dot1, dot2);

      return false;
    }

    const double lambda = (lambda_n / lambda_d);

    // estimate the update step delta_p
    error = (lambda * templateZM - imageWarped);
    __project_onto_jacobian(jacobian, error, errorProjection);
    deltaP = hessianInv * errorProjection;

//    t2 = get_time();
//    CF_DEBUG("PROJECT: %g ms", t2 - t1);
    //

    //
//    t1 = get_time();
    // update warping matrix
    __update_warp_matrix_x2y2(warp, deltaP);

//    t2 = get_time();
//    CF_DEBUG("UPDATE: %g ms", t2 - t1);
    //
  }


  return true;
}



// trans_x = -(dstSize.width / 2 - srcSize.width / 2);
// trans_y = -(dstSize.height / 2 - srcSize.height / 2);
void create_x2y2_grid(struct X2Y2Grid * dst, Size size, const Point & trans)
{
  Mat Xcoord = Mat(1, size.width, CV_32F);
  Mat Ycoord = Mat(size.height, 1, CV_32F);
  float * p;

  dst->Xgrid = Mat(size.height, size.width, CV_32F);
  dst->Ygrid = Mat(size.height, size.width, CV_32F);

  p = Xcoord.ptr<float>(0);
  for ( int j = 0; j < size.width; j++ ) {
    p[j] = (float) (j + trans.x);
  }

  p = Ycoord.ptr<float>(0);
  for (int j = 0; j < size.height; j++) {
    p[j] = (float) (j + trans.y);
  }

  repeat(Xcoord, size.height, 1, dst->Xgrid);
  repeat(Ycoord, 1, size.width, dst->Ygrid);

  multiply(dst->Xgrid, dst->Ygrid, dst->XYgrid);
  multiply(dst->Xgrid, dst->Xgrid, dst->X2grid);
  multiply(dst->Ygrid, dst->Ygrid, dst->Y2grid);
}

bool create_x2y2_remap(const struct X2Y2Grid & grid, const Mat & warp, Mat & map_x, Mat & map_y)
{
  // x' =  a00 * x + a01 * y + a02 + a03 * x * y + a04 * x*x + a05 * y*y
  // y' =  a10 * x + a11 * y + a12 + a13 * x * y + a14 * x*x + a15 * y*y

  double M[2][6];

  if ( warp.rows != 2 || warp.cols != 6 ) {
    CF_FATAL("Invalid warp matrix size specified: %dx%d. 2x6 is expected", warp.rows, warp.cols);
    return false;
  }

  if ( warp.type() != CV_32F && warp.type() != CV_64F ) {
    CF_FATAL("Invalid warp matrix type specified: %d. CV_32F or CV_64F is expected", warp.type());
    return false;
  }


  Mat matM(2, 6, CV_64F, (double *) M);
  const int nx = grid.Xgrid.cols, ny = grid.Xgrid.rows;

  warp.convertTo(matM, matM.type());

  if ( map_x.rows != ny || map_x.cols != nx ) {
    map_x = Mat(ny, nx, CV_32FC1, M[0][2]);
  }
  else {
    map_x.setTo(Scalar(M[0][2]));
  }

  scaleAdd(grid.Xgrid, M[0][0], map_x, map_x);
  scaleAdd(grid.Ygrid, M[0][1], map_x, map_x);
  scaleAdd(grid.XYgrid, M[0][3], map_x, map_x);
  scaleAdd(grid.X2grid, M[0][4], map_x, map_x);
  scaleAdd(grid.Y2grid, M[0][5], map_x, map_x);

  if ( map_y.rows != ny || map_y.cols != nx ) {
    map_y = Mat(ny, nx, CV_32FC1, M[1][2]);
  }
  else {
    map_y.setTo(Scalar(M[1][2]));
  }

  scaleAdd(grid.Xgrid, M[1][0], map_y, map_y);
  scaleAdd(grid.Ygrid, M[1][1], map_y, map_y);
  scaleAdd(grid.XYgrid, M[1][3], map_y, map_y);
  scaleAdd(grid.X2grid, M[1][4], map_y, map_y);
  scaleAdd(grid.Y2grid, M[1][5], map_y, map_y);

  return true;
}


bool x2y2_remap(const struct X2Y2Grid & g, const cv::Mat & src, cv::Mat & dst, const cv::Mat & M, int interp)
{
  Mat map_x, map_y;
  if ( !create_x2y2_remap(g, M, map_x, map_y) ) {
    CF_FATAL("create_x2y2_remap() fails");
    return false;
  }
  remap(src, dst, map_x, map_y, interp);
  return true;
}


bool x2y2_remap(const Mat & src, Mat & dst, const Mat & M, Size dstSize, int interp,
    const Point & trans /*= Point(0, 0)*/)
{
  struct X2Y2Grid g;
  Mat map_x, map_y;
  create_x2y2_grid(&g, dstSize, trans);
  return x2y2_remap(g, src, dst, M, interp);
}



template<class T>
static Mat affine2x2y2_(const Mat & M)
{
  Mat m;
  if ( M.rows == 2 && M.cols == 3 ) {
    m = Mat_<float>(2, 6, (float) (0));
    for ( uint i = 0; i < 2; ++i ) {
      for ( uint j = 0; j < 3; ++j ) {
        m.ptr<float>(i)[j] = M.ptr<const T>(i)[j];
      }
    }
  }
  return m;
}

Mat affine2x2y2(const Mat & M)
{
  switch ( M.type() ) {
    case CV_32FC1 :
    return affine2x2y2_<float>(M);
    case CV_64FC1 :
    return affine2x2y2_<double>(M);
    default :
    CF_FATAL("FATAL: Unhandled matrix type %d", M.type());
    break;
  }
  return Mat();
}

bool drawAndSaveKeypoints(const string & fname, InputArray image, const vector<KeyPoint> & keypoints, InputArray mask)
{
  Mat outImage;

  if ( image.empty() ) {
    CF_FATAL("Empty image specified to draw key points as '%s'", fname.c_str());
    return false;
  }

  normalize(image, outImage, 0, 255, NORM_MINMAX, CV_8UC3, mask);
  drawKeypoints(outImage, keypoints, outImage, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

  if ( !__save_image(outImage, fname) ) {
    CF_FATAL("[%s] __save_image() fails", fname.c_str());
    return false;
  }

  return true;
}

bool drawAndSaveMatches(const string & fname, InputArray image1, const vector<KeyPoint> & keypoints1, InputArray mask1,
    InputArray image2, const vector<KeyPoint> & keypoints2, InputArray mask2,
    const std::vector<DMatch> & matches1to2)
{
  Mat srcImg1, srcImg2, outImg;

  if ( image1.empty() || image2.empty() ) {
    CF_FATAL("Empty image specified to save as '%s'", fname.c_str());
    return false;
  }

  normalize(image1, srcImg1, 0, 255, NORM_MINMAX, CV_8UC3, mask1);
  normalize(image2, srcImg2, 0, 255, NORM_MINMAX, CV_8UC3, mask2);

  drawMatches(srcImg1, keypoints1, srcImg2, keypoints2, matches1to2, outImg, Scalar::all(-1), Scalar::all(-1),
      vector<char>(), DrawMatchesFlags::DEFAULT);

  if ( !__save_image(outImg, fname) ) {
    CF_FATAL("[%s] __save_image() fails", fname.c_str());
    return false;
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////

RetinaAlignBase :: RetinaAlignBase()
{
}

RetinaAlignBase :: ~RetinaAlignBase()
{
}

bool RetinaAlignBase :: run(const RetinaAlignBaseOpts & opts)
{
  Ptr<Feature2D> detector;
  void * ctx = 0;
  size_t n;
  int max_threads;
  Status master_status;

  opts_ = opts;


  if ( (max_threads = opts.max_threads()) < 1 ) {
    max_threads = std::thread::hardware_concurrency();
  }

  if ( !get_master_frame(master_name, master_frame, master_mask, &ctx) ) {
    CF_FATAL("get_master_frame() fails");
    return false;
  }

  if ( (detector = __create_keypoints_detector()).empty() ) {
    CF_FATAL("__create_keypoints_detector() fails");
    return false;
  }

  n = __extract_keypoints(detector, master_frame, master_mask, &master_hp, &master_hp_mask, &master_keypoints, &master_descriptors);
  CF_DEBUG("[%s] %zu master key points", master_name.c_str(), n);
  on_master_points_extracted(ctx, master_name, master_hp, master_hp_mask, master_keypoints, master_descriptors);
  if ( n < 3 ) {
    CF_FATAL("[%s] __extract_keypoints() returns %zu only key points", master_name.c_str(), n);
    CF_FATAL("[%s] master frame is not usable", master_name.c_str());
    return false;
  }

  // master assumed always self-aligned


  master_status.M = affine2x2y2((Mat) Mat::eye(2, 3, CV_32FC1));
  master_status.rho = 1;
  master_status.iterations = 0;
  onfinished(ctx, master_name, master_frame, master_mask, master_status);

  for ( int i = 0; i < max_threads; ++i ) {
    threads.emplace_back(std::thread(&RetinaAlignBase::worker_thread, this));
  }

  for ( size_t i = 0; i < threads.size(); ++i ) {
    threads[i].join();
  }

  threads.clear();

  return true;
}


void RetinaAlignBase :: worker_thread()
{
  Ptr<Feature2D> detector;
  BFMatcher matcher(NORM_HAMMING2, true);

  string fname;
  Mat frame, mask;
  void * ctx = 0;

  bool fOk = true;

  while ( get_next_frame(fname, frame, mask, &ctx) ) {

    Mat hp, hp_mask;
    vector<KeyPoint> frame_keypoints;
    Mat frame_descriptors;
    vector<Point2f> master_points, frame_points;
    vector<DMatch> all_matches;
    vector<DMatch> good_matches;

    Status S;
    size_t n;

    int maxIters = 20;
    double eps = 5e-4;

    double t1, t2;

    // process input

    t1 = __get_time();

    fOk = false;

    if ( detector.empty() && (detector = __create_keypoints_detector()).empty() ) {
      CF_FATAL("[%s] __create_keypoints_detector() fails", fname.c_str());
      break;
    }

    n = __extract_keypoints(detector, frame, mask, &hp, &hp_mask, &frame_keypoints, &frame_descriptors);
    CF_DEBUG("[%s] %zu key points", fname.c_str(), n);

    on_frame_points_extracted(ctx, fname, hp, hp_mask, frame_keypoints, frame_descriptors);

    if ( n < 3 ) {
      CF_FATAL("[%s] not enough key points (%zu), frame is not usable", fname.c_str(), n);
      goto end;
    }

    __match_key_points(matcher, master_descriptors, frame_descriptors, &all_matches);
    CF_DEBUG("[%s] %zu matches", fname.c_str(), all_matches.size());

    on_keypoints_matched(ctx, master_name, master_hp, master_hp_mask, master_keypoints,
        fname, hp, mask, frame_keypoints, all_matches);

    if ( all_matches.size() < 3 ) {
      CF_FATAL("[%s] not enough matches (%zu), frame is not usable", fname.c_str(), all_matches.size());
      goto end;
    }

    S.M = __estimate_transform(master_keypoints, frame_keypoints, all_matches, &good_matches);

    on_transform_estimated(ctx, master_name, master_hp, master_hp_mask, master_keypoints,
        fname, hp, mask, frame_keypoints, good_matches, S.M);

    if ( S.M.empty() ) {
      CF_FATAL("[%s] __estimate_transform() fails", fname.c_str());
      goto end;
    }

    fOk = __ecc_align(master_hp, master_hp_mask, hp, hp_mask, S.M, maxIters, eps, &S.iterations, &S.rho);
    on_ecc_aligned(ctx, fname, hp, hp_mask, fOk, S);

    if ( !fOk ) {
      CF_FATAL("[%s] __ecc_align() fails", fname.c_str());
      goto end;
    }

    t2 = __get_time();

    CF_DEBUG("[%s] fOk=%d rho=%g iterations=%d TIME=%g msec", fname.c_str(), fOk, S.rho, S.iterations, t2 - t1);
    CF_DEBUG("\n");

    end:

    onfinished(ctx, fname, frame, mask, S);
  }

}


///////////////////////////////////////////////////////////////////////////////

const std::string RetinaAlignOpts::default_debug_path;

RetinaAlign :: RetinaAlign()
  : current_index_ (0),
    master_index_(0)
{
}

RetinaAlign :: ~RetinaAlign()
{
}


bool RetinaAlign :: run(const RetinaAlignOpts & opts /*= RetinaAlignOpts()*/)
{
  this->opts_ = opts;
  return Base::run(opts);
}


void RetinaAlign :: set_master_index(size_t idx)
{
  master_index_ = idx;
}

void RetinaAlign :: add_frame(const std::string & fname, const Mat & frame, const Mat & mask)
{
  mtx_.lock();
  fnames_.emplace_back(fname);
  frames_.emplace_back(frame);
  masks_.emplace_back(mask);
  stats_.emplace_back();
  mtx_.unlock();
}

size_t RetinaAlign::frames_count(void) const
{
  return frames_.size();
}

std::string & RetinaAlign::fname(size_t index)
{
  return fnames_[index];
}

Mat & RetinaAlign::frame(size_t index)
{
  return frames_[index];
}

Mat & RetinaAlign::mask(size_t index)
{
  return masks_[index];
}

RetinaAlign::Status & RetinaAlign::status(size_t index)
{
  return stats_[index];
}


bool RetinaAlign::get_master_frame(std::string & fname, Mat & frame, Mat & mask, void ** ctx)
{
  return master_index_ < frames_.size() ? getframe(master_index_, fname, frame, mask, ctx) : false;
}


bool RetinaAlign::get_next_frame(std::string & fname, Mat & frame, Mat & mask, void ** ctx) /* override */
{
  bool fok = false;

  mtx_.lock();
  if ( current_index_ == master_index_ ) {
    ++current_index_;
  }
  if ( current_index_ < frames_.size() && (fok = getframe(current_index_, fname, frame, mask, ctx)) ) {
    ++current_index_;
  }
  mtx_.unlock();

  return fok;
}

bool RetinaAlign::getframe(size_t index, std::string & fname, Mat & frame, Mat & mask, void ** ctx)
{
  fname = fnames_[index];
  frame = frames_[index];
  mask = masks_[index];
  * ctx = (void*)(index);
  return true;
}

void RetinaAlign::on_master_points_extracted(void * ctx, const std::string &fname, const cv::Mat & frame,
    const cv::Mat & mask, std::vector<cv::KeyPoint> & keypoints, const cv::Mat & descriptors)
{
  (void) frame;
  (void) mask;
  (void) keypoints;
  (void) descriptors;

  size_t index = (size_t) (ctx);
  CF_DEBUG("[%3zu][%s] ", index, fname.c_str());

  if ( !fname.empty() && opts_.debug_keypoints() && !opts_.debug_path().empty() ) {
    const string outname = __ssprintf("%s/keypoints/%s.png", opts_.debug_path().c_str(),
        __extract_file_name(fname).c_str());
    drawAndSaveKeypoints(outname, frame, keypoints, mask);
  }
}

void RetinaAlign::on_frame_points_extracted(void * ctx, const std::string & fname, const cv::Mat & frame,
    const cv::Mat & mask, std::vector<cv::KeyPoint> & keypoints, const cv::Mat & descriptors)
{
  (void) frame;
  (void) mask;
  (void) keypoints;
  (void) descriptors;

  size_t index = (size_t) (ctx);
  CF_DEBUG("[%3zu][%s] ", index, fname.c_str());

  if (  !fname.empty() && opts_.debug_keypoints() && !opts_.debug_path().empty()) {
    const string outname = __ssprintf("%s/keypoints/%s.png", opts_.debug_path().c_str(),
        __extract_file_name(fname).c_str());
    drawAndSaveKeypoints(outname, frame, keypoints, mask);
  }
}

void RetinaAlign::on_keypoints_matched(void * ctx, const std::string & master_name, const cv::Mat & master_frame,
    const cv::Mat & master_mask, const std::vector<cv::KeyPoint> & master_keypoints,
    const std::string & frame_name, const cv::Mat & frame, const cv::Mat & mask,
    const std::vector<cv::KeyPoint> & frame_keypoints,
    const std::vector<cv::DMatch> & matches)
{
  (void) master_name;
  (void) master_frame;
  (void) master_mask;
  (void) master_keypoints;
  (void) frame_name;
  (void) frame;
  (void) mask;
  (void) frame_keypoints;
  (void) matches;

  size_t index = (size_t) (ctx);
  CF_DEBUG("[%3zu][%s] ", index, frame_name.c_str());

  if ( !frame_name.empty() && opts_.debug_matches() && !opts_.debug_path().empty() ) {

    const string outname = __ssprintf("%s/all_matches/%s.png", opts_.debug_path().c_str(),
        __extract_file_name(frame_name).c_str());

    drawAndSaveMatches(outname, master_frame, master_keypoints, master_mask,
        frame, frame_keypoints, mask, matches);
  }

}

void RetinaAlign::on_transform_estimated(void * ctx, const std::string & master_name, const cv::Mat & master_frame,
    const cv::Mat & master_mask, const std::vector<cv::KeyPoint> & master_keypoints,
    const std::string & frame_name, const cv::Mat & frame, const cv::Mat & mask,
    const std::vector<cv::KeyPoint> & frame_keypoints,
    const std::vector<cv::DMatch> & matches,
    const cv::Mat & M)
{
  (void) ctx;
  (void) (master_name);
  (void) (master_frame);
  (void) (frame_name);
  (void) (M);

  if ( !frame_name.empty() && opts_.debug_transform_match() && !opts_.debug_path().empty() ) {

    size_t index = (size_t) (ctx);
    CF_DEBUG("[%3zu][%s] ", index, frame_name.c_str());

    const string outname = __ssprintf("%s/matches/%s.png", opts_.debug_path().c_str(),
        __extract_file_name(frame_name).c_str());

    drawAndSaveMatches(outname, master_frame, master_keypoints, master_mask,
        frame, frame_keypoints, mask, matches);
  }

}

void RetinaAlign::on_ecc_aligned(void * ctx, const std::string & fname, const cv::Mat & frame, const cv::Mat & mask,
    bool fOk, const Status & status)
{
  (void) (ctx);
  (void) (fname);
  (void) (frame);
  (void) (mask);
  (void) (fOk);
  (void) (status);
  //size_t index = (size_t) (ctx);
  //CF_DEBUG("[%3zu][%s] ", index, fname.c_str());
}

void RetinaAlign::onfinished(void * ctx, std::string & fname, cv::Mat & frame, cv::Mat & mask, const Status & status)
{
  (void) (fname);
  (void) (frame);
  (void) (mask);

  size_t index = (size_t) (ctx);
  stats_[index] = status;

  CF_DEBUG("[%3zu][%s] ", index, fname.c_str());
}


///////////////////////////////////////////////////////////////////////////////


const std::string IntraSeriesRetinaAlignOpts:: default_output_path;

IntraSeriesRetinaAlign :: IntraSeriesRetinaAlign()
    : output_factor_(IntraSeriesRetinaAlignOpts::default_output_factor)
{
}

IntraSeriesRetinaAlign :: ~IntraSeriesRetinaAlign()
{
}

bool IntraSeriesRetinaAlign::run(const IntraSeriesRetinaAlignOpts & opts)
{
  Size srcSize;
  Size dstSize(opts.output_width(), opts.output_height());

  getmaxframesize(&srcSize);
  if ( srcSize.width < 1 || srcSize.height < 1 ) {
    CF_FATAL("FATAL: IntraSeriesRetinaAlign: Can't determine input frame size");
    return false;
  }

  if ( dstSize.width < 1 ) {
    dstSize.width = srcSize.width;
  }

  if ( dstSize.height < 1 ) {
    dstSize.height = srcSize.height;
  }

  create_x2y2_grid(&g_, dstSize, Point(-(dstSize.width - srcSize.width) / 2, -(dstSize.height - srcSize.height) / 2));
  aligned_frames_.resize(frames_count());
  aligned_masks_.resize(frames_count());
  output_path_ = opts.output_path();
  output_factor_ = opts.output_factor();

  return Base::run(opts);
}

void IntraSeriesRetinaAlign :: getmaxframesize(Size * size)
{
  size->width = size->height = 0;
  for ( size_t i = 0, n = frames_count(); i < n; ++i ) {
    const Mat & frame = Base::frame(i);
    if ( frame.cols > size->width ) {
      size->width = frame.cols;
    }
    if ( frame.rows > size->height ) {
      size->height = frame.rows;
    }
  }
}

void IntraSeriesRetinaAlign :: onfinished(void * ctx, std::string & fname, cv::Mat & frame, cv::Mat & mask, const Status & S)
{
  Base::onfinished(ctx, fname, frame, mask, S);

  if ( !S.M.empty() ) {

    size_t index = (size_t) (ctx);
    Mat & aligned_frame = aligned_frames_[index];
    Mat & aligned_mask = aligned_masks_[index];

    Mat map_x, map_y;

    if ( !create_x2y2_remap(g_, S.M, map_x, map_y) ) {
      CF_FATAL("[%s] FATAL: IntraSeriesRetinaAlign: create_x2y2_remap() fails", fname.c_str());
      return;
    }

    if ( !mask.empty() ) {
      remap(mask, aligned_mask, map_x, map_y, INTER_NEAREST);
      __imerode(aligned_mask, 3);
    }

    if ( !frame.empty() ) {
      remap(frame, aligned_frame, map_x, map_y, INTER_CUBIC);
      if ( !mask.empty() ) {
        aligned_frame.setTo(0, ~aligned_mask);
      }
      if ( output_factor_ != 0 && output_factor_ != 1 ) {
        aligned_frame *= output_factor_;
      }
    }

    if ( !output_path_.empty() && !aligned_frame.empty() ) {

      Mat outimg;
      string outname;

      aligned_frame.convertTo(outimg, CV_16UC1);

      if ( (outname = __extract_file_name(fname)).empty() ) {
        outname = __ssprintf("frame.%03zu.tif", index + 1);
      }
      outname = __ssprintf("%s/%s", output_path_.c_str(),  outname.c_str());

      if ( !__save_image(outimg, outname) ) {
        CF_FATAL("[%s] __save_image() fails", outname.c_str());
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////////////
