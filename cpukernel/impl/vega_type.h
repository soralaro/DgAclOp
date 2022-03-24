#define CS_NONE 0
#define CS_RGB 1
#define CS_BGR 2
#define CS_UV  3
#define CS_VU  4
#define CS_YUV420P 6
#define PLANE_SHIFT 24
#define CS_SHIFT 16
#define UNIT_SHIFT 8
#define MAKE_TYPE(plane, cs, unit)  (((plane) << PLANE_SHIFT) | ((cs) << CS_SHIFT) | ((unit) << UNIT_SHIFT))

enum vega_matrix_type {
    Undefined = 0,
    BGRPacked = MAKE_TYPE(1, CS_BGR, 3),
    RGBPacked = MAKE_TYPE(1, CS_RGB, 3),
    BGRPlanar = MAKE_TYPE(3, CS_BGR, 1),
    RGBPlanar = MAKE_TYPE(3, CS_RGB, 1),
    YUV420P = MAKE_TYPE(3, CS_YUV420P, 1),
    NV12 = MAKE_TYPE(2, CS_UV, 1),
    NV21 = MAKE_TYPE(2, CS_VU, 1),
    Gray = MAKE_TYPE(1, CS_NONE, 1)
};
typedef struct {
    unsigned long long addr;
    unsigned  int w;
    unsigned  int h;
    unsigned  int s_w;
    unsigned  int s_h;
    unsigned  int roi_x;
    unsigned  int roi_y;
    unsigned  int roi_w;
    unsigned  int roi_h;
    unsigned  int type;
}VegaMatrix;