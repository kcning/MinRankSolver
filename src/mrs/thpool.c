/* thpool.c: implementation of thpool.h */

#include "thpool.h"

#include <stdlib.h>
#include <signal.h>
#include <pthread.h>            // requires -lpthread at link time
#include <unistd.h>
#include <errno.h>
#include <setjmp.h>
#include <assert.h>


/* ========================================================================
 * struct Threadpool definition
 * ======================================================================== */

typedef struct {
    void (*func) (void* arg);
    void* arg;
} ThreadpoolJob;

#define THREADPOOL_MAX_JOB_NUM  65536

typedef struct {
    uint64_t id; // id of the worker; also index of its Thread struct
                 // in the internal array of the Threadpool it belongs to
    pthread_t pthread;
    Threadpool* restrict pool;
    jmp_buf* restrict envp; // for proper cleanup on hard shutdown
} Thread;

struct Threadpool {
    pthread_mutex_t lock;
    pthread_cond_t worker_event;
    pthread_cond_t queue_not_full;
    pthread_cond_t th_all_idle;
    pthread_cond_t th_init_done;

    volatile int64_t shutdown;
    volatile int64_t pause;

    volatile int64_t jobqueue_len;
    volatile int64_t jobqueue_head;
    volatile int64_t jobqueue_tail;
    ThreadpoolJob jobqueue[THREADPOOL_MAX_JOB_NUM]; // circular queue

    int64_t init_capacity;
    volatile int64_t thnum_active;
    volatile int64_t thnum_alive;
    Thread threads[];
};

// Global handle to the threadpool; Necessary because it's not possible to pass
// extra arguments to signal handlers. Using a global handle restricts the
// number of Threadpool to only one. I.e. a process must not create multiple
// Threadpools.
static Threadpool* g_tp_handle;

/* ========================================================================
 * function implementations
 * ======================================================================== */

/* usage: given a threadpool, check if its jobqueue is full. This function
 *      may only be called when holding the mutex of the pool.
 * params:
 *      tp: ptr to Threadpool
 * return: true if full; false otherwise */
static inline bool
thpool_jobqueue_full(const Threadpool* tp) {
    return tp->jobqueue_len == THREADPOOL_MAX_JOB_NUM;
}

/* usage: given a threadpool, check if its jobqueue is empty. This function
 *      may only be called when holding the mutex of the pool.
 * params:
 *      tp: ptr to Threadpool
 * return: true if empty; false otherwise */
static inline bool
thpool_jobqueue_empty(const Threadpool* tp) {
    return tp->jobqueue_len == 0;
}

/* usage: given a threadpool and a job, store the job into its job queue. This
 *      function may only be called when holding the mutex of the pool, and
 *      when the job queue is not full.
 * params:
 *      1) tp: ptr to Threadpool
 *      2) job: ptr to struct ThreadpoolJob
 * return: void */
static inline void
thpool_job_enqueue(Threadpool* restrict tp, const ThreadpoolJob* restrict job) {
    assert(!thpool_jobqueue_full(tp));

    if(tp->jobqueue_head == -1)
        tp->jobqueue_head = 0;

    tp->jobqueue_tail = (tp->jobqueue_tail + 1) % THREADPOOL_MAX_JOB_NUM;
    tp->jobqueue[tp->jobqueue_tail] = *job;

    tp->jobqueue_len++;
    assert(0 <= tp->jobqueue_len);
    assert(tp->jobqueue_len <= THREADPOOL_MAX_JOB_NUM);
}

/* usage: given a threadpool, pop a job from its job queue. This function
 *      may only be called when holding the mutex of the pool, and when the job
 *      queue is not empty.
 * params:
 *      1) tp: ptr to Threadpool
 *      2) job: ptr to struct ThreadpoolJob; container for the job
 * return: void */
static inline void
thpool_job_dequeue(Threadpool* restrict tp, ThreadpoolJob* restrict job) {
    assert(tp->jobqueue_head != -1);
    assert(!thpool_jobqueue_empty(tp));

    *job = tp->jobqueue[tp->jobqueue_head];
    if(tp->jobqueue_head == tp->jobqueue_tail)
        tp->jobqueue_head = tp->jobqueue_tail = -1;
    else
        tp->jobqueue_head = (tp->jobqueue_head + 1) % THREADPOOL_MAX_JOB_NUM;

    tp->jobqueue_len--;
    assert(0 <= tp->jobqueue_len);
    assert(tp->jobqueue_len < THREADPOOL_MAX_JOB_NUM);
}

/* usage: given a threadpool, return the number of workers that are still alive
 *      in the pool.
 * params:
 *      1) tp: ptr to Threadpool
 * return: the number of workers on success; negative value on error */
int64_t
thpool_alive_worker_num(Threadpool* tp) {
    if(!tp)
        return Threadpool_null;

    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    volatile int64_t num = tp->thnum_alive;
    assert(tp->thnum_alive >= tp->thnum_active);

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    return num;
}

/* usage: signal handler for thpool_worker; on SIGUSR1, pause the thread */
static inline void
thpool_worker_sa_handler_pause(int sig_id) {
    (void) sig_id; // get rid of 'unused param' warning from compiler

    Threadpool* tp = g_tp_handle;
    if(!tp)
        return;

    // NOTE: cannot use a condition variable because only async-signal-safe
    // functions are allowed in a signal handler.
    while(tp->pause)
        sleep(1);
}

/* usage: signal handler for thpool_worker; on SIGUSR2, terminate the thread */
static inline void
thpool_worker_sa_handler_terminate(int sig_id) {
    (void) sig_id; // get rid of 'unused param' warning from compiler

    Threadpool* tp = g_tp_handle;
    if(!tp)
        return;

    // find the Thread data structure for this thread; just do
    // a linear search because there're not so many threads.
    pthread_t own_pid = pthread_self();
    int64_t idx = -1;
    for(int64_t i = 0; i < tp->init_capacity; ++i) {
        if(tp->threads[i].pthread == own_pid) {
            idx = i;
            break;
        }
    }

    assert(idx != -1);
    longjmp(*(tp->threads[idx].envp), 36); // NOTE: 2nd param can be anything
}

/* usage: internal worker that dequeue jobs from the jobqueue and execute them.
 * params:
 *      1) t: ptr to Thread storing info for the worker
 * return : void*, as per requirement of pthread */
static inline void*
thpool_worker(Thread* t) {
    // register signal handlers
    struct sigaction sig_info;
    sigset_t block_mask;
    sigemptyset(&block_mask);
    sigaddset(&block_mask, SIGUSR2); // block signals for hard shutdown
    sig_info.sa_mask = block_mask;
    sig_info.sa_flags = 0;
    sig_info.sa_handler = thpool_worker_sa_handler_pause;
    if(sigaction(SIGUSR1, &sig_info, NULL)) {
        return NULL;
    }

    sigemptyset(&block_mask);
    sigaddset(&block_mask, SIGUSR1); // block signals for pause
    sig_info.sa_mask = block_mask;
    sig_info.sa_handler = thpool_worker_sa_handler_terminate;
    if(sigaction(SIGUSR2, &sig_info, NULL)) {
        return NULL;
    }

    Threadpool* const tp = t->pool;

    if(pthread_mutex_lock(&tp->lock))
        return NULL;

    tp->thnum_alive++;

    if(pthread_cond_signal(&tp->th_init_done)) {
        tp->thnum_alive--;
        pthread_mutex_unlock(&tp->lock);
        return NULL;
    }

    assert(tp->thnum_alive >= 0);

    if(pthread_mutex_unlock(&tp->lock)) {
        tp->thnum_alive--;
        return NULL;
    }

    // set up a recovery point for hard shutdown
    jmp_buf env;
    t->envp = &env;
    int jmp_v = setjmp(env);

    if(jmp_v) { // return from signal handler for SIGUSR2
        // try to release the lock in case this thread currently owns it.  If the
        // thread is not the owner, the function should only return EPERM since the
        // lock is robust and should not trigger a deadlock. Note that this
        // cannot be performed inside the signal handler because
        // pthread_mutex_unlock() is not async-signal-safe.
        int rv = pthread_mutex_unlock(&tp->lock);
        if(rv && rv != EPERM)
            return NULL;
    }

    while(!jmp_v) {
        if(pthread_mutex_lock(&tp->lock)) {
            // cannot update thnum_alive before returning
            return NULL;
        }

        while(thpool_jobqueue_empty(tp) && !tp->shutdown) {
            if(pthread_cond_wait(&tp->worker_event, &tp->lock)) {
                tp->thnum_alive--;
                pthread_mutex_unlock(&tp->lock);
                return NULL;
            }
        }

        if(tp->shutdown) {
            if(pthread_mutex_unlock(&tp->lock)) {
                tp->thnum_alive--;
                return NULL;
            } else
                break;
        }

        if(thpool_jobqueue_full(tp) &&
           pthread_cond_signal(&tp->queue_not_full)) {
            tp->thnum_alive--;
            pthread_mutex_unlock(&tp->lock);
            return NULL;
        }

        ThreadpoolJob job; // small enough to use stack memory
        thpool_job_dequeue(tp, &job);
        assert(!thpool_jobqueue_full(tp));

        tp->thnum_active++;
        assert(tp->thnum_alive >= tp->thnum_active && tp->thnum_active >= 0);
        if(pthread_mutex_unlock(&tp->lock)) {
            tp->thnum_active--;
            tp->thnum_alive--;
            return NULL;
        }

        job.func(job.arg);

        if(pthread_mutex_lock(&tp->lock)) {
            // cannot update thnum_alive before returning
            return NULL;
        }

        if(0 == --(tp->thnum_active) && thpool_jobqueue_empty(tp) &&
           pthread_cond_signal(&tp->th_all_idle)) {
            tp->thnum_alive--;
            pthread_mutex_unlock(&tp->lock);
            return NULL;
        }

        assert(tp->thnum_alive >= tp->thnum_active && tp->thnum_active >= 0);

        if(pthread_mutex_unlock(&tp->lock)) {
            tp->thnum_alive--;
            return NULL;
        }
    }

    if(pthread_mutex_lock(&tp->lock))
        return NULL;

    tp->thnum_alive--;
    assert(tp->thnum_alive >= 0);

    // terminating anyway; no need to check return value
    pthread_mutex_unlock(&tp->lock);

    return NULL;
}

/* usage: subroutine of thpool_create(). Wait until threads created earlier are
 *      initialized, then perform a soft shutdown.
 * params:
 *      1) tp: ptr to Threadpool
 *      2) n: the number of threads that has been created
 * return: 0 on success; non-zero on error */
static inline int
thpool_worker_init_abort(Threadpool* tp, int64_t n) {
    if(!n)
        return 0;

    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    while(tp->thnum_alive < n) {
        if(pthread_cond_wait(&tp->th_init_done, &tp->lock)) {
            pthread_mutex_unlock(&tp->lock);
            return Threadpool_cond_fail;
        }
    }

    tp->shutdown = 1; // do a soft shutdown

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    // wake up all workers
    if(pthread_cond_broadcast(&tp->worker_event))
        return Threadpool_cond_fail;

    for(int64_t i = 0; i < n; ++i) {
        if(pthread_join(tp->threads[i].pthread, NULL))
            return Threadpool_join_fail;
    }

    assert(tp->thnum_alive == 0);
    assert(tp->thnum_active == 0);

    return 0;
}

/* usage: create and initialize a threadpool; A process may only create one
 *      Threadpool at a time.
 * params:
 *      1) n: number of worker threads to create in the threadpool
 * return: a ptr to Threadpool on success; NULL on error */
Threadpool*
thpool_create(int64_t n) {
    if(g_tp_handle) // there can only be 1 Threadpool at a time
        return NULL;

    if(n <= 0)
        return NULL;

    if(n > THREADPOOL_MAX_JOB_NUM)
        n = THREADPOOL_MAX_JOB_NUM; // higher thread count makes no sense

    Threadpool* tp = malloc(sizeof(Threadpool) + sizeof(Thread) * n);
    if(!tp)
        return NULL;

    pthread_mutexattr_t lock_attr; // use a robust lock
    if(pthread_mutexattr_init(&lock_attr)) {
        free(tp);
        return NULL;
    }

    if(pthread_mutexattr_setrobust(&lock_attr, PTHREAD_MUTEX_ROBUST) ||
       pthread_mutex_init(&tp->lock, &lock_attr)) {
        pthread_mutexattr_destroy(&lock_attr);
        free(tp);
        return NULL;
    }

    if(pthread_mutexattr_destroy(&lock_attr)) {
        pthread_mutex_destroy(&tp->lock);
        free(tp);
        return NULL;
    }

    if(pthread_cond_init(&tp->worker_event, NULL)) {
        pthread_mutex_destroy(&tp->lock);
        free(tp);
        return NULL;
    }

    if(pthread_cond_init(&tp->queue_not_full, NULL)) {
        pthread_cond_destroy(&tp->worker_event);
        pthread_mutex_destroy(&tp->lock);
        free(tp);
        return NULL;
    }

    if(pthread_cond_init(&tp->th_all_idle, NULL)) {
        pthread_cond_destroy(&tp->queue_not_full);
        pthread_cond_destroy(&tp->worker_event);
        pthread_mutex_destroy(&tp->lock);
        free(tp);
        return NULL;
    }

    if(pthread_cond_init(&tp->th_init_done, NULL)) {
        pthread_cond_destroy(&tp->th_all_idle);
        pthread_cond_destroy(&tp->queue_not_full);
        pthread_cond_destroy(&tp->worker_event);
        pthread_mutex_destroy(&tp->lock);
        free(tp);
        return NULL;
    }

    tp->shutdown = 0;
    tp->pause = 0;
    tp->jobqueue_len = 0;
    tp->jobqueue_head = -1;
    tp->jobqueue_tail = -1;

    tp->init_capacity = n;
    tp->thnum_active = 0;
    tp->thnum_alive = 0;

    // init threads
    for(int64_t i = 0; i < n; ++i) {
        tp->threads[i].id = i;
        tp->threads[i].pool = tp;
        if(pthread_create(&tp->threads[i].pthread, NULL,
                          (void* (*) (void*)) thpool_worker,
                          (void*) (tp->threads + i))) {
            if(thpool_worker_init_abort(tp, i)) {
                // NOTE: cannot release lock and condition vars in this case
                return NULL;
            }
            pthread_cond_destroy(&tp->th_init_done);
            pthread_cond_destroy(&tp->th_all_idle);
            pthread_cond_destroy(&tp->queue_not_full);
            pthread_cond_destroy(&tp->worker_event);
            pthread_mutex_destroy(&tp->lock);
            free(tp);
            return NULL;
        }
    }

    if(pthread_mutex_lock(&tp->lock)) {
        // fail to acquire the lock already, calling thpool_worker_init_abort()
        // makes no sense
        // NOTE: cannot destory the condition vars in this case
        return NULL;
    }

    while(tp->thnum_alive != n) {
        if(pthread_cond_wait(&tp->th_init_done, &tp->lock)) {
            // th_init_done malfunctions already, calling
            // thpool_worker_init_abort() makes no sense
            pthread_mutex_unlock(&tp->lock);
            return NULL;
        }
    }

    if(pthread_mutex_unlock(&tp->lock)) {
        // fail to unlock the lock already; cannot call
        // thpool_worker_init_abort()
        return NULL;
    }

    // update the global handle
    g_tp_handle = tp;
    return tp;
}

/* usage: given a threadpool, add a job to execute into the pool. This function
 *      blocks the calling thread if the job queue in the pool is full.
 * params:
 *      1) tp: ptr to Threadpool
 *      2) func: function ptr to execute whose signature is void (*) (void*)
 *      3) arg: ptr to void, which is the argument for the function ptr
 * return: 0 if success; non-zero on error */
int
thpool_add_job(Threadpool* restrict tp, void (*f) (void*), void* restrict arg) {
    if(!tp)
        return Threadpool_null;

    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    while(thpool_jobqueue_full(tp) && !tp->shutdown) {
        if(pthread_cond_wait(&tp->queue_not_full, &tp->lock)) {
            pthread_mutex_unlock(&tp->lock);
            return Threadpool_cond_fail;
        }
    }

    if(tp->shutdown) {
        if(pthread_mutex_unlock(&tp->lock))
            return Threadpool_lock_fail;
        return Threadpool_shutdown;
    }

    ThreadpoolJob job = { .func = f, .arg = arg };
    thpool_job_enqueue(tp, &job);

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    if(pthread_cond_signal(&tp->worker_event))
        return Threadpool_cond_fail;

    return 0;
}

/* usage: given a threadpool, clear all queued jobs.
 * params:
 *      1) tp: ptr to Threadpool
 * return: 0 on success; non-zero value on error */
int
thpool_clear_jobs(Threadpool* tp) {
    if(!tp)
        return Threadpool_null;

    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    tp->jobqueue_head = -1;
    tp->jobqueue_tail = -1;
    tp->jobqueue_len = 0;

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    if(pthread_cond_broadcast(&tp->queue_not_full)) {
        return Threadpool_cond_fail;
    }

    return 0;
}

/* usage: Given a threadpool; check if all queued jobs are done
 *      and all workers are idle
 * params:
 *      1) tp: ptr to Threadpool
 * return: 1 if idle; 0 if not; negative value on error */
int
thpool_idle(Threadpool* tp) {
    if(!tp)
        return Threadpool_null;

    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    int rv = thpool_jobqueue_empty(tp) && (tp->thnum_active == 0);

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    return rv;
}

/* usage: Given a threadpool, block the calling thread until all
 *      queued jobs are finished
 * params:
 *      1) tp: ptr to Threadpool
 * return: 0 on success; non-zero value on error */
int
thpool_wait_jobs(Threadpool* tp) {
    if(!tp)
        return Threadpool_null;

    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    while(!tp->shutdown && (!thpool_jobqueue_empty(tp) || tp->thnum_active)) {
        if(pthread_cond_wait(&tp->th_all_idle, &tp->lock)) {
            pthread_mutex_unlock(&tp->lock);
            return Threadpool_cond_fail;
        }
    }

    int rv = tp->shutdown ? Threadpool_shutdown : 0;

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    return rv;
}

/* usage: Given a threadpool, pause all its workers. This function does nothing
 *      if the threadpool has been shut down.
 * params:
 *      1) tp: ptr to threadpool
 * return: 0 on success; non-zero value on error */
int
thpool_pause(Threadpool* tp) {
    if(!tp)
        return Threadpool_null;

    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    volatile int64_t shutdown = tp->shutdown;

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    if(shutdown)
        return Threadpool_shutdown;

    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    tp->pause = 1; // NOTE: the workers do not modify it

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    for(int64_t i = 0; i < tp->init_capacity; ++i) {
        int rv = pthread_kill(tp->threads[i].pthread, SIGUSR1);
        if(rv == ESRCH) // the thread has joined
            continue;
        else if(rv)
            return Threadpool_kill_fail;
    }

    return 0;
}

/* usage: Given a threadpool, resume all its workers. This function does nothing
 *      if the threadpool has been shut down.
 * params:
 *      1) tp: ptr to threadpool
 * return: 0 on success; non-zero value on error */
int
thpool_resume(Threadpool* tp) {
    if(!tp)
        return Threadpool_null;

    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    volatile int64_t shutdown = tp->shutdown;

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    if(shutdown)
        return Threadpool_shutdown;

    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    tp->pause = 0; // NOTE: the workers do not modify it

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    return 0;
}

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
thpool_hard_shutdown(Threadpool* tp) {
    if(!tp)
        return Threadpool_null;

    // wake up paused workers (if any) first, otherwise SIGUSR2 will be blocked
    int rv = thpool_resume(tp);
    if(rv)
        return rv;

    // first try a soft shutdown
    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    if(tp->shutdown) { // already shutdown
        pthread_mutex_unlock(&tp->lock);
        return Threadpool_shutdown;
    }

    tp->shutdown = 1;

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    // wake up all workers and threads blocking on the pool
    if(pthread_cond_broadcast(&tp->worker_event) ||
       pthread_cond_broadcast(&tp->queue_not_full) ||
       pthread_cond_broadcast(&tp->th_all_idle))
        return Threadpool_cond_fail;

    // TODO: do we need this?
    if(usleep(5000)) // wait for (some) threads to terminate properly
        return Threadpool_sleep_fail;

    // check if there's still worker alive
    volatile int tnum = (int) thpool_alive_worker_num(tp);
    if(tnum < 0)
        return tnum;

    // then perform the hard shutdown if necessary
    if(tnum > 0) {
        for(int64_t i = 0; i < tp->init_capacity; ++i) {
            int rv = pthread_kill(tp->threads[i].pthread, SIGUSR2);
            if(rv == ESRCH) // the thread has joined
                continue;
            else if(rv)
                return Threadpool_kill_fail;
        }
    }

    for(int64_t i = 0; i < tp->init_capacity; ++i) {
        if(pthread_join(tp->threads[i].pthread, NULL)) {
            return Threadpool_join_fail;
        }
    }

    tp->thnum_active = 0;
    tp->thnum_alive = 0;

    return 0;
}

/* usage: Given a threadpool, wait until all the workers finish their current
 *      jobs, and then terminate workers in the threadpool.  The threadpool
 *      becomes inoperable after this function returns and the only valid
 *      function call left is thpool_destroy().
 *
 *      If the workers are paused and not yet resumed, this function will
 *      resume the workers before shutting them down.
 * params:
 *      1) tp: ptr to threadpool
 * return: 0 on success; non-zero value on error */
static inline int
thpool_soft_shutdown(Threadpool* tp) {
    if(!tp)
        return Threadpool_null;

    // wake up paused workers (if any) first
    int rv = thpool_resume(tp);
    if(rv)
        return rv;

    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    if(tp->shutdown) { // already shutdown
        pthread_mutex_unlock(&tp->lock);
        return Threadpool_shutdown;
    }

    tp->shutdown = 1;

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    // wake up all workers and threads blocking on the pool
    if(pthread_cond_broadcast(&tp->worker_event) ||
       pthread_cond_broadcast(&tp->queue_not_full) ||
       pthread_cond_broadcast(&tp->th_all_idle))
        return Threadpool_cond_fail;

    for(int64_t i = 0; i < tp->init_capacity; ++i) {
        if(pthread_join(tp->threads[i].pthread, NULL))
            return Threadpool_join_fail;
    }

    assert(tp->thnum_alive == 0);
    assert(tp->thnum_active == 0);

    return 0;
}

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
thpool_destroy(Threadpool* tp, bool type) {
    if(!tp)
        return Threadpool_null;

    if(pthread_mutex_lock(&tp->lock))
        return Threadpool_lock_fail;

    volatile int64_t shutdown = tp->shutdown;

    if(pthread_mutex_unlock(&tp->lock))
        return Threadpool_lock_fail;

    if(!shutdown) {
        int rv = type ? thpool_soft_shutdown(tp) : thpool_hard_shutdown(tp);
        if(rv)
            return rv;
    }

    assert(tp->shutdown == 1);
    assert(tp->thnum_alive == 0);
    assert(tp->thnum_active == 0);

    // NOTE: destroying locked mutex and condition var is undefined behavior
    if(pthread_mutex_destroy(&tp->lock) ||
       pthread_cond_destroy(&tp->worker_event) ||
       pthread_cond_destroy(&tp->queue_not_full) ||
       pthread_cond_destroy(&tp->th_all_idle) ||
       pthread_cond_destroy(&tp->th_init_done))
        return Threadpool_free_fail;

    g_tp_handle = NULL; // reset the global handle
    free(tp);
    return 0;
}
