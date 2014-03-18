// Floating type for the solver.
typedef float rkv_float_t;

// Callback types.
typedef void (*rkv_dxdt_t)(rkv_float_t t, rkv_float_t* x, rkv_float_t *dx, void*);
typedef void (*rkv_on_step_t)(rkv_float_t t, rkv_float_t* xt, void*);

// Compute and output.
bool RungeKuttaVerner(
    int N, rkv_float_t t0, rkv_float_t t1,            // Number of vars, t0, t1
    rkv_float_t *x0, rkv_dxdt_t dxdt,            // x0, dxdt(x, t)
    rkv_float_t TOL, rkv_float_t hmin, rkv_float_t hmax,
    rkv_float_t* x1,
    rkv_on_step_t on_step = 0,
    void* userdata = 0
);
