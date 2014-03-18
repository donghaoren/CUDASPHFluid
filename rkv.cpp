// The Runge-Kutta-Verner Method for solving:
// x'(t) = x(t, y)
// x(t0) = x0
// from t0 to t1.

#include <cmath>
#include <cstdio>
using namespace std;

#include "rkv.h"

// A context object with vector helpers.
struct rkv_context_t {
    int N;
    rkv_context_t(int N_) { N = N_; }
    rkv_float_t* vector_init() {
        return new rkv_float_t[N];
    }
    void vector_free(rkv_float_t *x) {
        delete [] x;
    }
    void cp(rkv_float_t* dest, rkv_float_t* src) {
        for(int i = 0; i < N; i++)
            dest[i] = src[i];
    }
    void mul(rkv_float_t* dest, rkv_float_t* src, rkv_float_t k) {
        for(int i = 0; i < N; i++)
            dest[i] = src[i] * k;
    }
};

// Compute and output.
bool RungeKuttaVerner(
    int N, rkv_float_t t0, rkv_float_t t1,            // Number of vars, t0, t1
    rkv_float_t *x0, rkv_dxdt_t dxdt,            // x0, dxdt(x, t)
    rkv_float_t TOL, rkv_float_t hmin, rkv_float_t hmax,
    rkv_float_t* x1,
    rkv_on_step_t on_step,
    void* userdata
) {
    bool ret = true;
    rkv_context_t ctx(N);
    rkv_float_t t = t0;
    rkv_float_t h = hmax;
    if(t + h > t1) h = t1 - t;
    rkv_float_t* w = ctx.vector_init();
    rkv_float_t* k1 = ctx.vector_init();
    rkv_float_t* k2 = ctx.vector_init();
    rkv_float_t* k3 = ctx.vector_init();
    rkv_float_t* k4 = ctx.vector_init();
    rkv_float_t* k5 = ctx.vector_init();
    rkv_float_t* k6 = ctx.vector_init();
    rkv_float_t* k7 = ctx.vector_init();
    rkv_float_t* k8 = ctx.vector_init();
    rkv_float_t* dx = ctx.vector_init();
    rkv_float_t* tmp = ctx.vector_init();
    rkv_float_t* w_new = ctx.vector_init();
    rkv_float_t* w_new_hat = ctx.vector_init();

    ctx.cp(w, x0);

    bool continue_evaluation = true;
    bool is_first = true;
    while(continue_evaluation) {
        // First compute k1 ~ k8.
        dxdt(t,w,dx,userdata);
        ctx.mul(k1, dx, h);
        for(int i = 0; i < N; i++)
            tmp[i] = w[i]+k1[i]/6;
        dxdt(t+h/6,tmp,dx,userdata);
        ctx.mul(k2, dx, h);
        for(int i = 0; i < N; i++)
            tmp[i] = w[i]+k1[i]*4/75+k2[i]*16/75;
        dxdt(t+h*4/15,tmp,dx,userdata);
        ctx.mul(k3, dx, h);
        for(int i = 0; i < N; i++)
            tmp[i] = w[i]+k1[i]*5/6-k2[i]*8/3+k3[i]*5/2;
        dxdt(t+h*2/3,tmp,dx,userdata);
        ctx.mul(k4, dx, h);
        for(int i = 0; i < N; i++)
            tmp[i] = w[i]-k1[i]*165/64+k2[i]*55/6-k3[i]*425/64+k4[i]*85/96;
        dxdt(t+h*5/6,tmp,dx,userdata);
        ctx.mul(k5, dx, h);
        for(int i = 0; i < N; i++)
            tmp[i] = w[i]+k1[i]*12/5-k2[i]*8+k3[i]*4015/612-k4[i]*11/36+k5[i]*88/255;
        dxdt(t+h,tmp,dx,userdata);
        ctx.mul(k6, dx, h);
        for(int i = 0; i < N; i++)
            tmp[i] = w[i]-k1[i]*8263/15000+k2[i]*124/75-k3[i]*643/680-k4[i]*81/250+k5[i]*2484/10625;
        dxdt(t+h/15,tmp,dx,userdata);
        ctx.mul(k7, dx, h);
        for(int i = 0; i < N; i++)
            tmp[i] = w[i]+k1[i]*3501/1720-k2[i]*300/43+k3[i]*297275/52632-k4[i]*319/2322+k5[i]*24068/84065+k7[i]*3850/26703;
        dxdt(t+h,tmp,dx,userdata);
        ctx.mul(k8, dx, h);
        for(int i = 0; i < N; i++) {
            w_new[i] = w[i]+k1[i]*13/160+k3[i]*2375/5984+k4[i]*5/16+k5[i]*12/85+k6[i]*3/44;
            w_new_hat[i] = w[i]+k1[i]*3/40+k3[i]*875/2244+k4[i]*23/72+k5[i]*264/1955+k7[i]*125/11592+k8[i]*43/616;
        }
        rkv_float_t R = fabs(w_new[0] - w_new_hat[0]);;
        for(int i = 1; i < N; i++) {
            // Find the min error.
            rkv_float_t Ri = fabs(w_new[i] - w_new_hat[i]);
            if(R < Ri) R = Ri;
        }
        if(R <= TOL) {
            t = t + h;
            ctx.cp(w, w_new);
            if(on_step) on_step(t, w, userdata);
        }
        rkv_float_t delta = 0.84 * pow((double)TOL / R, 1.0/5.0); // fifth-order method.
        if(delta < 0.1) {
            h = 0.1 * h;
        } else if(delta > 4) {
            h = 4 * h;
        } else h = delta * h;
        if(h > hmax) h = hmax;
        if(t >= t1) {
            continue_evaluation = 0;
        } else if(t + h > t1) {
            h = t1 - t;
        } else if(h < hmin) {
            continue_evaluation = 0;
            printf("RKVError: minimum h reached, but still can't make error within TOL.\n");
            ret = false;
        }
    }

    // We're at t1.
    ctx.cp(x1, w);

    ctx.vector_free(w);
    ctx.vector_free(k1);
    ctx.vector_free(k2);
    ctx.vector_free(k3);
    ctx.vector_free(k4);
    ctx.vector_free(k5);
    ctx.vector_free(k6);
    ctx.vector_free(k7);
    ctx.vector_free(k8);
    ctx.vector_free(dx);
    ctx.vector_free(tmp);
    ctx.vector_free(w_new);
    ctx.vector_free(w_new_hat);

    return ret;
}

