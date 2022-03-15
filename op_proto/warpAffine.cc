#include "warpAffine.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(WarpAffineInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(WarpAffine, WarpAffineVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(WarpAffine, WarpAffineInferShape);
VERIFY_FUNC_REG(WarpAffine, WarpAffineVerify);

}  // namespace ge
