/*
 * retina-align.h
 *
 *  Created on: Jun 27, 2017
 *      Author: amyznikov
 */

#pragma once

#ifndef __retina_align_h__
#define __retina_align_h__

#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <mutex>
#include <thread>


struct X2Y2Grid {
  cv::Mat Xgrid, Ygrid, XYgrid, X2grid, Y2grid;
};


// trans_x = -(dstSize.width / 2 - srcSize.width / 2);
// trans_y = -(dstSize.height / 2 - srcSize.height / 2);
void create_x2y2_grid(struct X2Y2Grid * dst, cv::Size size, const cv::Point & trans = cv::Point(0, 0));

// x' =  a00 * x + a01 * y + a02 + a03 * x * y + a04 * x*x + a05 * y*y
// y' =  a10 * x + a11 * y + a12 + a13 * x * y + a14 * x*x + a15 * y*y
bool create_x2y2_remap(const struct X2Y2Grid & grid, const cv::Mat & M, cv::Mat & map_x, cv::Mat & map_y);


// trans_x = -(dstSize.width / 2 - srcSize.width / 2);
// trans_y = -(dstSize.height / 2 - srcSize.height / 2);
bool x2y2_remap(const cv::Mat & src, cv::Mat & dst, const cv::Mat & M, cv::Size dstSize, int interp,
    const cv::Point & trans = cv::Point(0, 0));

bool x2y2_remap(const struct X2Y2Grid & g, const cv::Mat & src, cv::Mat & dst, const cv::Mat & M,
    int interp = cv::INTER_LINEAR);


cv::Mat affine2x2y2(const cv::Mat & M);

bool drawAndSaveKeypoints(const std::string & fname, cv::InputArray image, const std::vector<cv::KeyPoint> & keypoints,
    cv::InputArray mask = cv::noArray());

bool drawAndSaveMatches(const std::string & fname, cv::InputArray image1, const std::vector<cv::KeyPoint> & keypoints1,
    cv::InputArray mask1, cv::InputArray image2, const std::vector<cv::KeyPoint> & keypoints2, cv::InputArray mask2,
    const std::vector<cv::DMatch> & matches1to2);


class RetinaAlignBaseOpts {

public:

  static double constexpr default_align_precision = 4e-5;
  static double constexpr default_max_iterations = 20;
  static int constexpr default_max_threads = -1;

  RetinaAlignBaseOpts(void)
      : align_precision_(default_align_precision),
        max_iterations_(default_max_iterations),
        max_threads_(default_max_threads)
  {
  }

  double align_precision(void) const {
    return align_precision_;
  }

  bool set_align_precision(double v) {
    align_precision_ = v;
    return true;
  }

  int max_iterations(void) const {
    return max_iterations_;
  }

  bool set_max_iterations(int v) {
    max_iterations_ = v;
    return true;
  }

  int max_threads(void) const {
    return max_threads_;
  }

  bool set_max_threads(int v) {
    max_threads_ = v;
    return true;
  }

private:
  double align_precision_;
  int max_iterations_;
  int max_threads_;
};


class RetinaAlignBase {

public:

  struct Status {
    cv::Mat M;
    double rho;
    int iterations;
    Status() : rho(-1), iterations(-1) {}
    Status(const cv::Mat & _M, double _rho = -1, int _iterations = 0) : M(_M), rho(_rho), iterations(_iterations) {}
  };

  RetinaAlignBase();
  virtual ~RetinaAlignBase();
  bool run(const RetinaAlignBaseOpts & opts = RetinaAlignBaseOpts());

protected:
  virtual bool get_master_frame(std::string & fname, cv::Mat & frame, cv::Mat & mask, void ** ctx) = 0;
  virtual bool get_next_frame(std::string & fname, cv::Mat & frame, cv::Mat & mask, void ** ctx) = 0;


  // event signal
  virtual void on_master_points_extracted(void * ctx, const std::string &fname, const cv::Mat & frame,
      const cv::Mat & mask, std::vector<cv::KeyPoint> & keypoints, const cv::Mat & descriptors)
  {
    (void) ctx;
    (void) fname;
    (void) frame;
    (void) mask;
    (void) keypoints;
    (void) descriptors;
  }

  // event signal
  virtual void on_frame_points_extracted(void * ctx, const std::string & fname, const cv::Mat & frame,
      const cv::Mat & mask, std::vector<cv::KeyPoint> & keypoints, const cv::Mat & descriptors)
  {
    (void) ctx;
    (void) fname;
    (void) frame;
    (void) mask;
    (void) keypoints;
    (void) descriptors;
  }

  // event signal
  virtual void on_keypoints_matched(void * ctx, const std::string & master_name, const cv::Mat & master_frame,
      const cv::Mat & master_mask, const std::vector<cv::KeyPoint> & master_keypoints,
      const std::string & frame_name, const cv::Mat & frame, const cv::Mat & mask,
      const std::vector<cv::KeyPoint> & frame_keypoints,
      const std::vector<cv::DMatch> & matches)
  {
    (void) ctx;
    (void) master_name;
    (void) master_frame;
    (void) master_mask;
    (void) master_keypoints;
    (void) frame_name;
    (void) frame;
    (void) mask;
    (void) frame_keypoints;
    (void) matches;
  }

  // event signal
  virtual void on_transform_estimated(void * ctx, const std::string & master_name, const cv::Mat & master_frame,
      const cv::Mat & master_mask, const std::vector<cv::KeyPoint> & master_keypoints,
      const std::string & frame_name, const cv::Mat & frame, const cv::Mat & mask,
      const std::vector<cv::KeyPoint> & frame_keypoints,
      const std::vector<cv::DMatch> & matches, const cv::Mat & M)
  {
    (void) ctx;
    (void) master_name;
    (void) master_frame;
    (void) master_mask;
    (void) master_keypoints;
    (void) frame_name;
    (void) frame;
    (void) mask;
    (void) frame_keypoints;
    (void) matches;
    (void) (M);
  }

  // event signal
  virtual void on_ecc_aligned(void * ctx, const std::string & fname, const cv::Mat & frame, const cv::Mat & mask,
      bool fOk, const Status & status)
  {
    (void) (ctx);
    (void) (fname);
    (void) (frame);
    (void) (mask);
    (void) (fOk);
    (void) (status);
  }

  virtual void onfinished(void * ctx, std::string & fname, cv::Mat & frame, cv::Mat & mask, const Status & status)
  {
    (void) (ctx);
    (void) (fname);
    (void) (frame);
    (void) (mask);
    (void) (status);
  }

private:
  void worker_thread();

private:
  RetinaAlignBaseOpts opts_;
  std::vector<std::thread> threads;
  std::string master_name;
  cv::Mat master_frame, master_mask;
  cv::Mat master_hp, master_hp_mask;
  std::vector<cv::KeyPoint> master_keypoints;
  cv::Mat master_descriptors;

};



class RetinaAlignOpts
    : public RetinaAlignBaseOpts
{
public:

  typedef RetinaAlignBaseOpts
      Base;

  static const std::string default_debug_path;
  static constexpr bool default_debug_keypoints = false;
  static constexpr bool default_debug_matches = false;
  static constexpr bool default_debug_transform_match = false;

  RetinaAlignOpts()
    : debug_path_(default_debug_path),
      debug_keypoints_(default_debug_keypoints),
      debug_matches_(default_debug_matches),
      debug_transform_match_(default_debug_transform_match)

  {
  }

  const std::string & debug_path(void) const {
    return debug_path_;
  }

  bool set_debug_path(const std::string & v) {
    debug_path_ = v;
    return true;
  }

  bool debug_keypoints(void) const  {
    return debug_keypoints_;
  }

  bool set_debug_keypoints(bool v) {
    debug_keypoints_ = v;
    return true;
  }

  bool debug_matches(void) const  {
    return debug_matches_;
  }

  bool set_debug_matches(bool v) {
    debug_matches_ = v;
    return true;
  }

  bool debug_transform_match(void) const  {
    return debug_transform_match_;
  }

  bool set_debug_transform_match(bool v) {
    debug_transform_match_ = v;
    return true;
  }

private:
  std::string debug_path_;
  bool debug_keypoints_;
  bool debug_matches_;
  bool debug_transform_match_;
};


class RetinaAlign
    : public RetinaAlignBase
{
public:
  typedef RetinaAlignBase
      Base;

  typedef Base::Status
      Status;

  RetinaAlign();
  ~RetinaAlign();

  void set_master_index(size_t idx);

  void add_frame(const std::string & fname, const cv::Mat & frame, const cv::Mat & mask);
  size_t frames_count(void) const;
  std::string & fname(size_t index);
  cv::Mat & frame(size_t index);
  cv::Mat & mask(size_t index);
  Status & status(size_t index);

  bool run(const RetinaAlignOpts & opts = RetinaAlignOpts());

protected:
  bool get_master_frame(std::string & fname, cv::Mat & frame, cv::Mat & mask, void ** ctx) override;
  bool get_next_frame(std::string & fname, cv::Mat & image, cv::Mat & mask, void ** ctx) override;


  // event handlers

  void on_master_points_extracted(void * ctx, const std::string &fname, const cv::Mat & frame,
      const cv::Mat & mask, std::vector<cv::KeyPoint> & keypoints, const cv::Mat & descriptors)  override;

  void on_frame_points_extracted(void * ctx, const std::string & fname, const cv::Mat & frame,
      const cv::Mat & mask, std::vector<cv::KeyPoint> & keypoints, const cv::Mat & descriptors) override;

  void on_keypoints_matched(void * ctx, const std::string & master_name, const cv::Mat & master_frame,
      const cv::Mat & master_mask, const std::vector<cv::KeyPoint> & master_keypoints,
      const std::string & frame_name, const cv::Mat & frame, const cv::Mat & mask,
      const std::vector<cv::KeyPoint> & frame_keypoints,
      const std::vector<cv::DMatch> & matches) override;

  void on_transform_estimated(void * ctx, const std::string & master_name, const cv::Mat & master_frame,
      const cv::Mat & master_mask, const std::vector<cv::KeyPoint> & master_keypoints,
      const std::string & frame_name, const cv::Mat & frame, const cv::Mat & mask,
      const std::vector<cv::KeyPoint> & frame_keypoints,
      const std::vector<cv::DMatch> & matches,
      const cv::Mat & M) override;

  void on_ecc_aligned(void * ctx, const std::string & fname, const cv::Mat & frame, const cv::Mat & mask,
      bool fOk, const Status & status) override;

  void onfinished(void * ctx, std::string & fname, cv::Mat & frame, cv::Mat & mask, const Status & status) override;


private:
  bool getframe(size_t index, std::string & fname, cv::Mat & image, cv::Mat & mask, void ** ctx);

private:
  RetinaAlignOpts opts_;
  std::mutex mtx_;
  std::vector<std::string> fnames_;
  std::vector<cv::Mat> frames_;
  std::vector<cv::Mat> masks_;
  std::vector<Status> stats_;
  size_t current_index_;
  size_t master_index_;
};


class IntraSeriesRetinaAlignOpts
    : public RetinaAlignOpts
{
public:

  typedef RetinaAlignOpts
      Base;

  static const std::string default_output_path;
  static constexpr int default_output_width = 0;
  static constexpr int default_output_height = 0;
  static constexpr double default_output_factor = 16;

  IntraSeriesRetinaAlignOpts()
      : output_path_(default_output_path),
        output_width_(default_output_width),
        output_height_(default_output_height),
        output_factor_(default_output_factor)
  {
  }

  const std::string & output_path(void) const {
    return output_path_;
  }

  bool set_output_path(const std::string & v) {
    output_path_ = v;
    return true;
  }

  int output_width(void) const {
    return output_width_;
  }

  bool set_output_width(int v) {
    output_width_ = v;
    return true;
  }

  int output_height(void) const  {
    return output_height_;
  }

  bool set_output_height(int v) {
    output_height_ = v;
    return true;
  }


  double output_factor(void) const  {
    return output_factor_;
  }

  bool set_output_factor(double v) {
    output_factor_ = v;
    return true;
  }

private:
  std::string output_path_;
  int output_width_;
  int output_height_;
  double output_factor_;
};


class IntraSeriesRetinaAlign
    : public RetinaAlign
{
public:
  typedef RetinaAlign
      Base;

  typedef Base::Status
      Status;

  IntraSeriesRetinaAlign();
  ~IntraSeriesRetinaAlign();

  bool run(const IntraSeriesRetinaAlignOpts & opts = IntraSeriesRetinaAlignOpts());

protected:
  void onfinished(void * ctx, std::string & fname, cv::Mat & frame, cv::Mat & mask, const Status & status) override;
  void getmaxframesize(cv::Size * size);

private:
  std::string output_path_;
  double output_factor_;
  std::vector<cv::Mat> aligned_frames_;
  std::vector<cv::Mat> aligned_masks_;
  cv::Size srcSize_, dstSize_;
  X2Y2Grid g_;
};



#endif /* __retina_align_h__ */

