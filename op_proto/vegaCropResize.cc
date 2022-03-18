#include "vegaCropResize.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(VegaCropResizeInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(VegaCropResize, VegaCropResizeVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(VegaCropResize, VegaCropResizeInferShape);
VERIFY_FUNC_REG(VegaCropResize, VegaCropResizeVerify);

}  // namespace ge
