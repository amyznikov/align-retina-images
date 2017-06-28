/*
 * align-retina-images
 *
 *  Created on: Jun 22, 2017
 *      Author: amyznikov
 */
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include "retina-align.h"
#include "aperture.h"
#include "debug.h"


using namespace cv;
using namespace std;


# define DEFAULT_MKDIR_MODE \
    (S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH)



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
      if ( mkdir(tmp, mode) != 0 && errno != EEXIST ) {
        return false;
      }
      *p = '/';
    }
  }

  return mkdir(tmp, mode) == 0 || errno == EEXIST ? true : false;
}

static bool __load_image(Mat * image, const string & fname, int pixtype /* = -1*/)
{
  CF_DEBUG("C imread('%s')", fname.c_str());

  if ( !(*image = imread(fname, IMREAD_UNCHANGED)).data ) {
    CF_CRITICAL("imread(%s) fails", fname.c_str());
    return false;
  }

  if ( pixtype != -1 && image->type() != pixtype ) {
    image->convertTo(*image, pixtype);
  }

  return true;
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


template<class T>
static bool __get_bounding_rect_(const Mat & src, Rect * rc)
{
  int left = INT_MAX, top = INT_MAX, right = -1, bottom = -1;

  for ( int i = 0;  i <  src.rows; ++i ) {
    const T * p = src.ptr<const T>(i);
    for ( int j = 0;  j <  src.cols; ++j ) {
      if ( p[j] ) {
        if ( j < left ) {
          left = j;
        }
        if ( j > right ) {
          right = j;
        }
        if ( i < top ) {
          top = i;
        }
        if ( i > bottom ) {
          bottom = i;
        }
      }
    }
  }

  rc->x = left < INT_MAX ? left : 0;
  rc->y = top < INT_MAX ? top : 0;
  rc->width = ((left < INT_MAX && right >= 0 ? right - left + 1 : 0));
  rc->height = ((top < INT_MAX && bottom >= 0 ? bottom - top + 1 : 0));

  return true;
}

static bool __get_bounding_rect(const Mat & src, Rect * rc)
{
  switch ( src.type() ) {
    case CV_8UC1 :
    return __get_bounding_rect_<uint8_t>(src, rc);
    case CV_8SC1 :
    return __get_bounding_rect_<int8_t>(src, rc);
    case CV_16UC1 :
    return __get_bounding_rect_<uint16_t>(src, rc);
    case CV_16SC1 :
    return __get_bounding_rect_<int16_t>(src, rc);
    case CV_32SC1 :
    return __get_bounding_rect_<int32_t>(src, rc);
    case CV_32FC1 :
    return __get_bounding_rect_<float>(src, rc);
    case CV_64FC1 :
    return __get_bounding_rect_<double>(src, rc);
    default :
    CF_FATAL("Unsupported matrix type %d encountered", src.type());
    break;
  }
  return false;
}


int main(int argc, char *argv[])
{

  struct X2Y2Grid g;
  Mat map_x, map_y;
  Size srcSize;
  Size dstSize;

  RetinaAlignOpts opts;
  RetinaAlign a;

  cf_set_logfilename("stderr");
  cf_set_loglevel(CF_LOG_DEBUG);

  /*
   * Load images and extract auto aperture masks
   * */
  for ( int i = 1; i < argc; ++i ) {
    Mat image, mask;
    if ( !__load_image(&image, argv[i], CV_32FC1) ) {
      CF_CRITICAL("load_image() fails for '%s'", argv[i]);
    }
    else {
      if ( !extract_auto_aperture(image, &mask, 15, 1, true) ) {
        mask = Mat(image.size(), CV_8UC1, 255);
      }
      a.add_frame(__extract_file_name(argv[i]), image, mask);
    }
  }

  if ( a.frames_count() < 2 ) {
    fprintf(stderr, "Need at least 2 images to proceed.\n"
        "Use:\n"
        "  align-retina-images <fname1> <fname2> [fname3 ...]\n");
    return 1;
  }


  /**
   * Get max frame size
   */
  for ( size_t i = 0, n = a.frames_count(); i < n; ++i ) {
    const Mat & frame = a.frame(i);
    if ( frame.cols > srcSize.width ) {
      srcSize.width = frame.cols;
    }
    if ( frame.rows > srcSize.height ) {
      srcSize.height = frame.rows;
    }
  }
  dstSize = srcSize * 5 / 2;


  /*
   * Compute remap to master frame
   * */
  opts.set_debug_keypoints(true);
  opts.set_debug_transform_match(true);
  opts.set_debug_path("./example");
  a.set_master_index(0);
  a.run(opts);



  /*
   * Remap and save all frames
   * */
  create_x2y2_grid(&g, dstSize, Point(-(dstSize.width - srcSize.width), -(dstSize.height - srcSize.height)) / 2);


  Mat master;
  Mat master_mask;

  for ( size_t i = 0, n = a.frames_count(); i < n; ++i ) {
    RetinaAlign::Status s = a.status(i);
    Mat image, mask, weight;

    if ( s.M.empty() ) {
      CF_FATAL("[%s] : emty warp matrix", a.fname(i).c_str());
      continue;
    }

    create_x2y2_remap(g, s.M, map_x, map_y);

    remap(a.frame(i), image, map_x, map_y, INTER_CUBIC);
    remap(a.mask(i), mask, map_x, map_y, INTER_NEAREST);
    image.setTo(0, ~mask);

    if ( i == 0 ) { // master
      master = image;
      master_mask = mask;
      continue;
    }

    bitwise_and(master_mask, mask, weight);
    weight.convertTo(weight, CV_32FC1, 0.5 / 255);
    weight.setTo(1, weight == 0 );

    add(master, image, image);
    multiply(image, weight, image);

    bitwise_or(master_mask, mask, mask);

    Rect rc;
    __get_bounding_rect(mask, &rc);
    image = image(rc);

    normalize(image, image, 0, 255, NORM_MINMAX, CV_8UC1); // , mask
    __save_image(image, __ssprintf("./%s/aligned/%s.png", opts.debug_path().c_str(), a.fname(i).c_str()));
  }

  return 0;
}


