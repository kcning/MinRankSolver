/* thpool.h: header file for struct Threadpool */

#ifndef  __THREADPOOL_H__
#define  __THREADPOOL_H__

#include <stdint.h>
#include <stdbool.h>

typedef struct Threadpool Threadpool;

typedef enum {
   Threadpool_lock_fail = -1,
   Threadpool_cond_fail = -2,
   Threadpool_join_fail = -3,
   Threadpool_kill_fail = -4,
   Threadpool_null = -5,
   Threadpool_free_fail = -6,
   Threadpool_sleep_fail = -7,
   Threadpool_shutdown = 1,
} ThreadpoolErrorType;

/* ========================================================================
 * function prototypes
 * ======================================================================== */

/* usage: create and initialize a threadpool; A process may only create one
 *      Threadpool at a time.
 * params:
 *      1) n: number of worker threads to create in the threadpool
 * return: a ptr to Threadpool on success; NULL on error */
Threadpool*
thpool_create(int64_t n);

/* usage: given a threadpool, return the number of workers that are still alive
 *      in the pool.
 * params:
 *      1) tp: ptr to Threadpool
 * return: the number of workers on success; negative value on error */
int64_t
thpool_alive_worker_num(Threadpool* tp);

/* usage: given a threadpool, add a job to execute into the pool. This function
 *      blocks the calling thread if the job queue in the pool is full.
 * params:
 *      1) tp: ptr to Threadpool
 *      2) func: function ptr to execute whose signature is void (*) (void*)
 *      3) arg: ptr to void, which is the argument for the function ptr
 * return: 0 if success; non-zero on error */
int
thpool_add_job(Threadpool* restrict tp, void (*f) (void*), void* restrict arg);

/* usage: given a threadpool, clear all queued jobs.
 * params:
 *      1) tp: ptr to Threadpool
 * return: 0 on success; non-zero value on error */
int
thpool_clear_jobs(Threadpool* tp);

/* usage: Given a threadpool; check if all queued jobs are done
 *      and all workers are idle
 * params:
 *      1) tp: ptr to Threadpool
 * return: 1 if idle; 0 if not; negative value on error */
int
thpool_idle(Threadpool* tp);

/* usage: Given a threadpool, block the calling thread until all
 *      queued jobs are finished
 * params:
 *      1) tp: ptr to Threadpool
 * return: 0 on success; non-zero value on error */
int
thpool_wait_jobs(Threadpool* tp);

/* usage: Given a threadpool, pause all its workers. This function does nothing
 *      if the threadpool has been shut down.
 * params:
 *      1) tp: ptr to threadpool
 * return: 0 on success; non-zero value on error */
int
thpool_pause(Threadpool* tp);

/* usage: Given a threadpool, resume all its workers. This function does nothing
 *      if the threadpool has been shut down.
 * params:
 *      1) tp: ptr to threadpool
 * return: 0 on success; non-zero value on error */
int
thpool_resume(Threadpool* tp);

/* usage: Given a threadpool, kill all its workers. Note that there might be
 *      unexecuted jobs left in the pool and the states of resouces (e.g. mutex
 *      and condition variable) accessed in the job can be undefined.
 *
 *      If this function encounter errors when killing the workers, it returns a
 *      non-zero value and may leave some workers alive.
 *
 *      The threadpool becomes inoperable after this function returns and the
 *      only valid function call left is thpool_destroy().
 *
 *      If the workers are paused and not yet resumed, this function will
 *      resume the workers before killing them.
 * params:
 *      1) tp: ptr to threadpool
 * return : 0 on success; non-zero value on error */
int
thpool_hard_shutdown(Threadpool* tp);

/* usage: Given a threadpool, terminate all its workers and release its
 *      resources. The terminination can be soft or hard:
 *
 *      soft: wait until all workers finish their current jobs
 *      hard: terminate the workers immediately without waiting
 *
 *      Hard terminination might leave system resources accessed or allocated
 *      in user supplied jobs in an undefined state, and should be used as a
 *      last resort.
 *
 *      This function will simply release resources of the threadpool if a
 *      shutdown has been performed.
 * params:
 *      1) tp: ptr to Threadpool
 *      2) type: false for hard terminination, true for soft terminination
 * return: 0 on success; non-zero value on error */
int
thpool_destroy(Threadpool* tp, bool type);

#endif // __THREADPOOL_H__
