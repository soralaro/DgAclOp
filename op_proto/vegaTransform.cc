#include "vegaTransform.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(VegaTransformInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(VegaTransform, VegaTransformVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(VegaTransform, VegaTransformInferShape);
VERIFY_FUNC_REG(VegaTransform, VegaTransformVerify);

}  // namespace ge
