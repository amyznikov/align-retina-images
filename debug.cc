/*
 * debug.c
 *
 *  Created on: Aug 27, 2016
 *      Author: amyznikov
 */

#include "debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <mutex>

#ifdef _WIN32
// FIXME: provide alternatives for:
//  clock_gettime(CLOCK_REALTIME)
//  syscall (SYS_gettid);
//  backtrace()
//  typedef uint32_t pid_t;
#else
# include <sys/syscall.h>
# include <ucontext.h>
# include <execinfo.h>
#endif


/* Standard  terminal colors */
#define TCFG_GRAY     "\033[30m" //  set foreground color to gray
#define TCFG_RED      "\033[31m" //  set foreground color to red
#define TCFG_GREEN    "\033[32m" //  set foreground color to green
#define TCFG_YELLOW   "\033[33m" //  set foreground color to yellow
#define TCFG_BLUE     "\033[34m" //  set foreground color to blue
#define TCFG_PURPLE   "\033[35m" //  set foreground color to magenta (purple)
#define TCFG_CYAN     "\033[36m" //  set foreground color to cyan
#define TCFG_WHITE    "\033[37m" //  set foreground color to white

#define TCBG_GRAY     "\033[40m" //  set background color to gray
#define TCBG_RED      "\033[41m" //  set background color to red
#define TCBG_GREEN    "\033[42m" //  set background color to green
#define TCBG_YELLOW   "\033[43m" //  set background color to yellow
#define TCBG_BLUE     "\033[44m" //  set background color to blue
#define TCBG_PURPLE   "\033[45m" //  set background color to magenta (purple)
#define TCBG_CYAN     "\033[46m" //  set background color to cyan
#define TCBG_WHITE    "\033[47m" //  set background color to white

#define TC_RESET      "\033[0m"  //  reset; clears all colors and styles (to white on black)
#define TC_BOLD       "\033[1m"  //  bold on
#define TC_ITALIC     "\033[3m"  //  italics on
#define TC_UNDERLINE  "\033[4m"  //  underline on
#define TC_BLINK      "\033[5m"  //  blink on
#define TC_REVERSE    "\033[7m"  //  reverse video on
#define TC_INVISIBLE  "\033[8m"  //  nondisplayed (invisible)

//\033[x;yH moves cursor to line x, column y
//\033[xA moves cursor up x lines
//\033[xB moves cursor down x lines
//\033[xC moves cursor right x spaces
//\033[xD moves cursor left x spaces
//\033[2J clear screen and home cursor

// current time
struct ctime {
  int year;
  int month;
  int day;
  int hour;
  int min;
  int sec;
  int msec;
};


static std::mutex mtx;
static void (*logfunc)(void * context, const char * msg) = NULL;
static void * logcontext = NULL;
static FILE * fplog = NULL;
static char * logfilename = NULL;
static uint32_t logmask = CF_LOG_DEBUG;



static inline pid_t gettid() {
#ifdef _WIN32
  return 0; // fixme: return process id under win32
#else
  return (pid_t) syscall (SYS_gettid);
#endif
}

static void getctime(struct ctime * ct)
{
  struct timespec t;
  struct tm * tm;

  clock_gettime(CLOCK_REALTIME, &t);
  tm = gmtime(&t.tv_sec);

  ct->year = tm->tm_year + 1900;
  ct->month = tm->tm_mon + 1;
  ct->day = tm->tm_mday;
  ct->hour = tm->tm_hour;
  ct->min = tm->tm_min;
  ct->sec = tm->tm_sec;
  ct->msec = t.tv_nsec / 1000000;
}

static const char * getctime_string(char buf[32])
{
  struct ctime ct;
  getctime(&ct);
  snprintf(buf, 31, "%.4d-%.2d-%.2d-%.2d:%.2d:%.2d.%.3d",
      ct.year, ct.month, ct.day, ct.hour, ct.min, ct.sec, ct.msec);
  return buf;
}

/*
 * Color debug lines for Linux terminals,
 * depending on message severity level
 */
static inline void plogbegin(int pri)
{
#ifdef _WIN32
  (void)(pri);
#else
  const char * ctrl = NULL;
  switch ( pri ) {
    case CF_LOG_FATAL:
    case CF_LOG_CRITICAL:
      ctrl = TCFG_RED TC_BOLD;
    break;
  case CF_LOG_ERROR:
    ctrl = TCFG_RED;
    break;
  case CF_LOG_WARNING:
    ctrl = TCFG_YELLOW;
    break;
  case CF_LOG_NOTICE:
      ctrl = TC_BOLD;
    break;
  //case CF_LOG_INFO:
  case CF_LOG_DEBUG:
      ctrl = TCFG_GREEN;
    break;
  default:
    break;
  }

  if ( ctrl ) {
    fputs(ctrl, fplog);
  }
#endif
}

static inline void plogend(int pri)
{
#ifdef _WIN32
  (void)(pri);
#else
  const char * ctrl = NULL;
  switch ( pri )
  {
  case CF_LOG_FATAL:
    case CF_LOG_CRITICAL:
    case CF_LOG_ERROR:
    case CF_LOG_WARNING:
    case CF_LOG_NOTICE:
    //case CF_LOG_INFO:
    case CF_LOG_DEBUG:
    ctrl = TC_RESET;
    break;
  default:
    break;
  }

  if ( ctrl ) {
    fputs(ctrl, fplog), fflush(fplog);
  }
#endif
}



void cf_set_logfunc(void (*func)(void * context, const char * msg), void * context)
{
  mtx.lock();
  logfunc = func;
  logcontext = context;
  mtx.unlock();
}

void cf_set_logfile(FILE * fp)
{
  mtx.lock();

  if ( fplog && fplog != stderr && fplog != stdout ) {
    fclose(fplog), fplog = NULL;
  }
  fplog = fp;

  mtx.unlock();

}

FILE * cf_get_logfile(void)
{
  return fplog;
}


bool cf_set_logfilename(const char * fname)
{
  bool fok = false;

  mtx.lock();

  free(logfilename), logfilename = NULL;

  if ( fplog && fplog != stderr && fplog != stdout ) {
    fclose(fplog), fplog = NULL;
  }

  if ( !fname ) {
    fok = true;
  }
  else if ( !(logfilename = strdup(fname)) ) {
    fprintf(stderr, "fatal error: strdup(logfilename) fails: %s\n", strerror(errno));
  }
  else if ( strcasecmp(fname, "stderr") == 0 ) {
    fplog = stderr;
  }
  else if ( strcasecmp(fname, "stdout") == 0 ) {
    fplog = stdout;
  }
  else if ( !(fplog = fopen(logfilename, "a")) ) {
    fprintf(stderr, "fatal error: strdup(logfilename) fails: %s\n", strerror(errno));
  }
  else {
    char ctime_string[32];
    getctime_string(ctime_string);
    fprintf(fplog, "\n\nNEW LOG STARTED AT %s\n", ctime_string);
    fok = true;
  }

  mtx.unlock();

  return fok;
}

const char * cf_get_logfilename(void)
{
  return logfilename;
}

void cf_set_loglevel(uint32_t mask)
{
  logmask = mask;
}

uint32_t cf_get_loglevel(void)
{
  return logmask;
}

static char pric(int pri)
{
  char ch;

  switch ( pri ) {
    case CF_LOG_FATAL :
      ch = 'F';
    break;
    case CF_LOG_CRITICAL :
      ch = 'C';
    break;
    case CF_LOG_ERROR :
      ch = 'E';
    break;
    case CF_LOG_WARNING :
      ch = 'W';
    break;
    case CF_LOG_NOTICE :
      ch = 'N';
    break;
    case CF_LOG_INFO :
      ch = 'I';
    break;
    case CF_LOG_DEBUG :
      ch = 'D';
    break;
    default :
      ch = 'U';
    break;
  }
  return ch;
}

static void do_plogv(int pri, const char * func, int line, const char * format, va_list arglist)
{
  char ctime_string[32];
  getctime_string(ctime_string);


  if ( logfunc ) {

    char buf[8192] = "";
    int n;

    n = snprintf(buf, sizeof(buf) - 1, "|%c|%6d|%s| %-28s(): %4d :", pric(pri), (int) gettid(), ctime_string, func, line);

    if ( n > 0 && n < (int) (sizeof(buf) - 1) ) {
      vsnprintf(buf + n, sizeof(buf) - 1 - n, format, arglist);
    }

    if ( fplog ) {
      fputs(buf, fplog);
      fputc('\n', fplog);
      fflush(fplog);
    }

    logfunc(logcontext, buf);
  }
  else if ( fplog ) {
    //plogbegin(pri & 0x07);
    fprintf(fplog, "|%c|%6d|%s| %-28s(): %4d :", pric(pri),  (int)  gettid(), ctime_string, func, line);
    vfprintf(fplog, format, arglist);
    fputc('\n', fplog);
    fflush(fplog);
    //plogend(pri & 0x07);
  }

}

void cf_plogv(int pri, const char * func, int line, const char * format, va_list arglist)
{
  mtx.lock();
  if ( (fplog || logfunc) && (pri & 0x07) <= (logmask & 0x07) ) {
    do_plogv(pri, func, line, format, arglist);
  }
  mtx.unlock();
}

void cf_plog(int pri, const char * func, int line, const char * format, ...)
{
  mtx.lock();
  if ( (fplog || logfunc) && (pri & 0x07) <= (logmask & 0x07) ) {
    va_list arglist;
    va_start(arglist, format);
    do_plogv(pri, func, line, format, arglist);
    va_end(arglist);
  }
  mtx.unlock();
}



/**
 * Dump back trace into log file
 */
void cf_pbt(void)
{
#ifdef _WIN32
    // fixme: backtrace call under win32
#else
  int size;
  void * array[256];
  char ** messages = NULL;

  mtx.lock();

  if ( fplog ) {

    size = backtrace(array, sizeof(array) / sizeof(array[0]));
    messages = backtrace_symbols(array, size);

    for ( int i = 0; i < size; ++i ) {
      fprintf(fplog, "[bt]: (%d) %p %s\n", i, array[i], messages[i]);
    }
  }

  mtx.unlock();

  if ( messages ) {
    free(messages);
  }
#endif
}



#ifndef _WIN32

/**
 * Custom signal handler
 */
static void my_signal_handler(int signum, siginfo_t *si, void * context)
{
  int ignore = 0;
  int status = 0;
  const ucontext_t * uc = (ucontext_t *) context;
  void * caller_address;

#if ( __aarch64__ )
  caller_address = (void *) uc->uc_mcontext.pc;
#elif ( __arm__)
  caller_address = (void *) uc->uc_mcontext.arm_pc;
#else
  caller_address = (void *) uc->uc_mcontext.gregs[16]; // REG_RIP
#endif

  if ( signum != SIGWINCH ) {
    CF_CRITICAL("SIGNAL %d (%s)", signum, strsignal(signum));
  }

  switch ( signum ) {
    case SIGINT :
    case SIGQUIT :
    case SIGTERM :
      status = 0;
    break;

    case SIGSEGV :
    case SIGSTKFLT :
    case SIGILL :
    case SIGBUS :
    case SIGSYS :
    case SIGFPE :
    case SIGABRT :
      status = EXIT_FAILURE;
      CF_FATAL("Fault address:%p from %p", si->si_addr, caller_address);
      CF_PBT();
    break;

    default :
      ignore = 1;
    break;
  }

  if ( !ignore ) {
    exit(status);
  }
}


/**
 * setup_signal_handler()
 *    see errno on failure
 */
bool cf_setup_signal_handler(void)
{
  struct sigaction sa;
  int sig;

  memset(&sa, 0, sizeof(sa));

  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = my_signal_handler;

  for ( sig = 1; sig <= SIGUNUSED; ++sig ) {
    /* skip unblockable signals */
    if ( sig != SIGKILL && sig != SIGSTOP && sig != SIGCONT && sigaction(sig, &sa, NULL) != 0 ) {
      return false;
    }
  }

  return true;
}

#endif
