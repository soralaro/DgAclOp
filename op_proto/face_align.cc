#include "face_align.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(FaceAlignInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(FaceAlign, FaceAlignVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FaceAlign, FaceAlignInferShape);
VERIFY_FUNC_REG(FaceAlign, FaceAlignVerify);

}  // namespace ge
