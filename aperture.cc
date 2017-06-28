/*
 * aperture.cc
 *
 *  Created on: Nov 23, 2016
 *      Author: amyznikov
 */


#include "aperture.h"
#include "debug.h"


using namespace std;
using namespace cv;
///////////////////////////////////////////////////////////////////////////////



#define DUMP_HIST 0
#if DUMP_HIST
# define DBG_HIST CF_DEBUG
#else
# define DBG_HIST(...)
#endif



#if DUMP_HIST


static void dumphist(FILE * fp, const Mat & gshist, double minv, double step)
{
  fprintf(fp, "I\tV\tH\n");
  for ( int i = 0; i < gshist.cols; ++i ) {
    fprintf(fp, "%4d\t%9g\t%9g\n", i, minv + i * step, histVal(gshist, i));
  }
}

static void dumphist(const char * fname, const Mat & gshist, double minv, double step)
{
  FILE * fp;

  if ( strcasecmp(fname, "stdout") == 0 ) {
    fp = stdout;
  }
  else if ( strcasecmp(fname, "stderr") == 0 ) {
    fp = stderr;
  }
  else if ( !(fp = fopen(fname, "w")) ){
    CF_CRITICAL("fopen('%s') fails: %s", fname, strerror(errno));
  }

  if ( fp ) {
    dumphist(fp, gshist, minv, step);
    if ( fp != stderr && fp != stdout ) {
      fclose(fp);
    }
  }
}
#endif



static inline void imerode(Mat & image, int size, enum MorphShapes shape = MORPH_RECT)
{
  cv::erode(image, image, getStructuringElement(shape, Size(size, size)));
}

static inline void imerode(InputArray src, OutputArray dst, int size, enum MorphShapes shape = MORPH_RECT)
{
  cv::erode(src, dst, getStructuringElement(shape, Size(size, size)));
}

static void create_noise_map(const Mat & src, Mat & dst, InputArray _mask = noArray())
{
  /*
   * Estimate the standard deviation of the noise in a gray-scale image.
   *  J. Immerkr, Fast Noise Variance Estimation,
   *    Computer Vision and Image Understanding,
   *    Vol. 64, No. 2, pp. 300-302, Sep. 1996
   *
   * Matlab code:
   *  https://www.mathworks.com/matlabcentral/fileexchange/36941-fast-noise-estimation-in-images
   */

  Mat M, H;

  /* Compute sum of absolute values of laplacian */
  static const float K[3*3] = {
      1, -2,  1,
     -2,  4, -2,
      1, -2,  1
  };

  filter2D(src, H, CV_32FC1, Mat(3, 3, CV_32FC1, (void*) K));
  absdiff(H, 0, H);

  M = _mask.getMat();
  if ( !M.empty() ) {
    imerode(M, M, 5);
    H.setTo(0, ~M);
  }

  dst = H;
}

static double estimate_noise(const Mat & src, InputArray _mask = noArray())
{
  Mat M, H;
  int N;
  double sigma;

  create_noise_map(src, H);

  if ( !(M = _mask.getMat()).empty() ) {
    imerode(M, M, 5);
    H.setTo(0, ~M);
  }

  sigma = sum(H).val[0];

  if ( !isfinite(sigma) ) {
    CF_FATAL("BAD SIGMA=%g", sigma);
  }

  // scale sigma with proposed coefficients
  if ( !M.empty() ) {
    N = countNonZero(M);
  }
  else {
    N = H.size().area();
  }

  return sigma * sqrt(M_PI_2) / (6.0 * N);
}


static void mk_cumulative_hist(Mat & gshist)
{
  double sum = 0;
  float * h;

  h = gshist.ptr<float>(0);
  for ( int i = 0; i < gshist.cols; ++i ) {
    sum += h[i];
    h[i] = (float) (sum);
  }
}

bool calc_hist(const Mat & image, Mat & gshist, int hbins, bool cummulative,
    bool scaled, double * minv, double * maxv, InputArray mask)
{
  double noise = 0;

  if ( *minv >= *maxv || !isfinite(*minv) || !isfinite(*maxv) ) {
    minMaxLoc(image, minv, maxv, NULL, NULL, mask);
    if ( *minv >= *maxv || !isfinite(*minv) || !isfinite(*maxv) ) {
      CF_CRITICAL("invalid pixel ragne found : min=%g max=%g", *minv, *maxv);
      return false;
    }
  }

  if ( hbins < 1 ) {
    noise = estimate_noise(image);
    if ( noise <= 0 || !isfinite(noise) ) {
      CF_CRITICAL("estimate_noise() returns unexpected value %g", noise);
      return false;
    }
    hbins = min(65536, max((int) ((*maxv - *minv) / noise + 0.5), 2));
  }

  int histSize[] = { hbins };
  int channels[] = { 0 };
  float hrange[] = { (float) *minv, (float) *maxv + FLT_EPSILON };
  const float * ranges[] = { hrange };

  calcHist(&image, 1, channels, mask, gshist, 1, histSize, ranges, true, false);
  if ( gshist.empty() ) {
    CF_CRITICAL("calcHist() fails");
    return false;
  }

  if ( gshist.cols == 1 ) {
    transpose(gshist, gshist);
  }

  if ( scaled ) {
    divide(gshist, cv::sum(gshist), gshist);
  }

  if ( cummulative ) {
    mk_cumulative_hist(gshist);
  }

  return true;
}


static int nbzeros(const uint8_t * HB, int cmax)
{
  int nbz = 0;
  for ( int i = 0; i < cmax; ++i ) {
    nbz += !HB[i];
  }
  return nbz;
}

static bool findcomp(const float H[], const uint8_t HB[], int pos, int cmax, int * low1, int * up1, int * low0,
    int * up0, int * imax, int * imin)
{
  // skip zeros
  while ( pos < cmax && !HB[pos] ) {
    ++pos;
  }
  if ( pos == cmax ) {
    return false;
  }

  // 1 found
  *imax = *low1 = pos;
  while ( pos < cmax && HB[pos] ) {
    if ( H[pos] > H[*imax] ) {
      *imax = pos;
    }
    ++pos;
  }
  if ( pos == cmax ) {
    return false;
  }
  *up1 = pos - 1;

  // 0 found
  *imin = *low0 = pos;
  while ( pos < cmax && !HB[pos] ) {
    if ( H[pos] < H[*imin] ) {
      *imin = pos;
    }
    ++pos;
  }
  *up0 = pos - 1;

  return pos < cmax && HB[pos];
}


static bool searchcomps(const float H[], const uint8_t HB[], int cmax, int * threshpos)
{
  int low1, up1, low0, up0;
  int imax, imin;
  int pos = 0;

  while ( findcomp(H, HB, pos, cmax, &low1, &up1, &low0, &up0, &imax, &imin) ) {
    DBG_HIST("comp at pos=%d imax=%d imin=%d H[imax]=%g H[imin]=%g ratio=%g up0=%d", pos, imax, imin, H[imax], H[imin], H[imax] / H[imin], up0);
    if ( H[imin] <= FLT_EPSILON || H[imax] / H[imin] > 4 ) {
      *threshpos = imin;
      DBG_HIST("comp accepted at thresh=%d", * threshpos);
      return true;
    }
    pos = up0 + 1;
  }

  return false;
}


static int find_aperture_threshold(Mat & gsh)
{
  Mat gshb;
  Rect roi;
  int thresh = -1;

  double tol = 1e-4;
  bool all_zeros = false;

  for ( int cmax = gshb.cols / 4; cmax <= 3 * gshb.cols / 2; cmax += gshb.cols / 4 ) {

    bool found = false;
    int ccmax = min(cmax, gshb.cols);

    tol = 1e-4;
    all_zeros = false;

    while ( tol < 0.3 ) {

      threshold(gsh, gshb, tol, 255, THRESH_BINARY);
      gshb.convertTo(gshb, CV_8UC1);

      if ( nbzeros(gshb.ptr<const uint8_t>(0), ccmax) == ccmax ) {
        all_zeros = true;
        break;
      }

      if ( searchcomps(gsh.ptr<const float>(0), gshb.ptr<const uint8_t>(0), ccmax, &thresh) ) {
        if ( thresh < 3 * gshb.cols / 4 ) {
          DBG_HIST("FOUND AT TOL=%g", tol);
          found = true;
          break;
        }
      }

      tol += 1e-4;
    }

    if ( found ) {
      break;
    }
  }


  threshold(gsh, gshb, tol, 1, THRESH_BINARY);

#if DUMP_HIST
  dumphist("histb.txt", gshb, 0, 1);
#endif

  if ( all_zeros ) {
    CF_CRITICAL("FAIL COND: top limit reached: tol=%g", tol);
    return -1;
  }


  float * H = gsh.ptr<float>(0);

  double m = mean(gsh, gsh > tol).val[0];
  if ( H[thresh] > 0.1 * m ) {
    CF_ERROR("FAIL COND: H[thresh] > 0.1 * mean(gsh,gsh > tol) thresh=%d H[thresh]=%g mean(gsh, gsh > tol)=%g",
        thresh, H[thresh], m);
    return -1;
  }

  return thresh;
}


bool extract_auto_aperture(const Mat & image, Mat * mask, int erode_size, int border_size, bool da)
{
  Mat gsh, damask;
  double minv = 0, maxv = 0, step, t;
  int T;

  * mask = Mat();

  if ( da ) {
    imerode(image > FLT_EPSILON, damask, 9);
    draw_aperture_border(damask, 1);
  }

  if ( !calc_hist(image, gsh, 0, false, true, &minv, &maxv, damask) ) {
    CF_CRITICAL("calcGrayscaleHist() fails");
    return false;
  }

  GaussianBlur(gsh, gsh, Size(3,1), 1);

  step = (maxv - minv) / gsh.cols;
  DBG_HIST("minv=%g maxv=%g step=%g", minv, maxv, step);

#if DUMP_HIST
  dumphist("hist.txt", gsh, minv, step);
#endif


  T = find_aperture_threshold(gsh);
  DBG_HIST("T=%d (%g)", T, minv + T * step);

  if ( T <= 0 ) {
    CF_CRITICAL("Can't locate aperture threshold.\n"
        " minv=%g maxv=%g step=%g T=%d (%g)",
        minv, maxv, step, T, minv + T * step);
    return false;
  }

  t = minv + T * step;

  threshold(image, *mask, t, 255, THRESH_BINARY);
  mask->convertTo(*mask, CV_8UC1);
  medianBlur(*mask, *mask, 5);
  imerode(*mask, 3);

  if ( !damask.empty() ) {
    bitwise_and(*mask, damask, *mask);
  }

  if ( border_size > 0 ) {
    draw_aperture_border(*mask, border_size);
  }

  if ( erode_size ) {
    imerode(*mask, erode_size );
  }

  return true;
}

void draw_aperture_border(Mat & image, int size, const Scalar & color /*= Scalar(0)*/)
{
  if ( size > 0 ) {
    // top
    image(Rect(0, 0, image.cols - 1, size)) = color;

    // bottom
    image(Rect(0, image.rows - size, image.cols, size)) = color;

    // left
    image(Rect(0, size, size, image.rows - 2 * size)) = color;

    // right
    image(Rect(image.cols - size, size, size, image.rows - 2 * size)) = color;
  }
}

void draw_aperture_border(const Mat & src, Mat & dst, int size, const Scalar & color)
{
  dst = src.clone();
  draw_aperture_border(dst, size, color);
}

