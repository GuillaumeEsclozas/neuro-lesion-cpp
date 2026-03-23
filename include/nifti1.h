/*
 * nifti1.h
 * Stripped-down NIfTI-1 header definitions from nifti_clib.
 */

#ifndef NIFTI1_H
#define NIFTI1_H

#include <cstdint>

#define NII_SINGLE_MAGIC "n+1\0"
#define NII_PAIR_MAGIC   "ni1\0"

#define DT_UNKNOWN    0
#define DT_UINT8      2
#define DT_INT16      4
#define DT_INT32      8
#define DT_FLOAT32   16
#define DT_FLOAT64   64
#define DT_INT8     256
#define DT_UINT16   512
#define DT_UINT32   768

#define NIFTI_INTENT_NONE    0
#define NIFTI_INTENT_LABEL 1002

#define NIFTI_XFORM_UNKNOWN      0
#define NIFTI_XFORM_SCANNER_ANAT 1
#define NIFTI_XFORM_ALIGNED_ANAT 2
#define NIFTI_XFORM_TALAIRACH    3
#define NIFTI_XFORM_MNI_152      4

#define NIFTI_UNITS_MM   2
#define NIFTI_UNITS_SEC  8

#pragma pack(push, 1)
struct nifti_1_header {
    int32_t   sizeof_hdr;
    char      data_type[10];
    char      db_name[18];
    int32_t   extents;
    int16_t   session_error;
    char      regular;
    char      dim_info;

    int16_t   dim[8];
    float     intent_p1;
    float     intent_p2;
    float     intent_p3;
    int16_t   intent_code;
    int16_t   datatype;
    int16_t   bitpix;
    int16_t   slice_start;
    float     pixdim[8];
    float     vox_offset;
    float     scl_slope;
    float     scl_inter;
    int16_t   slice_end;
    char      slice_code;
    char      xyzt_units;
    float     cal_max;
    float     cal_min;
    float     slice_duration;
    float     toffset;
    int32_t   glmax;
    int32_t   glmin;

    char      descrip[80];
    char      aux_file[24];

    int16_t   qform_code;
    int16_t   sform_code;
    float     quatern_b;
    float     quatern_c;
    float     quatern_d;
    float     qoffset_x;
    float     qoffset_y;
    float     qoffset_z;
    float     srow_x[4];
    float     srow_y[4];
    float     srow_z[4];

    char      intent_name[16];
    char      magic[4];
};
#pragma pack(pop)

static_assert(sizeof(nifti_1_header) == 348, "NIfTI-1 header must be 348 bytes");

#endif /* NIFTI1_H */
