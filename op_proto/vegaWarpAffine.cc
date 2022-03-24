#include "vegaWarpAffine.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(VegaWarpAffineInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(VegaWarpAffine, VegaWarpAffineVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(VegaWarpAffine, VegaWarpAffineInferShape);
VERIFY_FUNC_REG(VegaWarpAffine, VegaWarpAffineVerify);

}  // namespace ge
