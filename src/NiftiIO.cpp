#include "NiftiIO.h"
#include "nifti1.h"
#include "znzlib.h"

#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace {

bool has_gz_extension(const std::string& path) {
    if (path.size() < 3) return false;
    return path.compare(path.size() - 3, 3, ".gz") == 0;
}

std::vector<float> convert_to_float(const void* raw, size_t nvox, int16_t datatype,
                                     float slope, float intercept)
{
    std::vector<float> out(nvox);

    if (slope == 0.0f) { slope = 1.0f; intercept = 0.0f; }

    switch (datatype) {
    case DT_FLOAT32: {
        auto* src = static_cast<const float*>(raw);
        for (size_t i = 0; i < nvox; i++) out[i] = src[i] * slope + intercept;
    } break;
    case DT_FLOAT64: {
        auto* src = static_cast<const double*>(raw);
        for (size_t i = 0; i < nvox; i++) out[i] = static_cast<float>(src[i] * slope + intercept);
    } break;
    case DT_INT16: {
        auto* src = static_cast<const int16_t*>(raw);
        for (size_t i = 0; i < nvox; i++) out[i] = src[i] * slope + intercept;
    } break;
    case DT_UINT16: {
        auto* src = static_cast<const uint16_t*>(raw);
        for (size_t i = 0; i < nvox; i++) out[i] = src[i] * slope + intercept;
    } break;
    case DT_INT32: {
        auto* src = static_cast<const int32_t*>(raw);
        for (size_t i = 0; i < nvox; i++) out[i] = static_cast<float>(src[i]) * slope + intercept;
    } break;
    case DT_UINT8: {
        auto* src = static_cast<const uint8_t*>(raw);
        for (size_t i = 0; i < nvox; i++) out[i] = src[i] * slope + intercept;
    } break;
    case DT_INT8: {
        auto* src = static_cast<const int8_t*>(raw);
        for (size_t i = 0; i < nvox; i++) out[i] = src[i] * slope + intercept;
    } break;
    case DT_UINT32: {
        auto* src = static_cast<const uint32_t*>(raw);
        for (size_t i = 0; i < nvox; i++) out[i] = static_cast<float>(src[i]) * slope + intercept;
    } break;
    default:
        throw std::runtime_error("Unsupported NIfTI datatype: " + std::to_string(datatype));
    }

    return out;
}

int bytes_per_voxel(int16_t datatype) {
    switch (datatype) {
    case DT_UINT8:   case DT_INT8:    return 1;
    case DT_INT16:   case DT_UINT16:  return 2;
    case DT_INT32:   case DT_UINT32:  case DT_FLOAT32: return 4;
    case DT_FLOAT64: return 8;
    default: return 0;
    }
}

} // anonymous namespace


NiftiVolume nifti::load(const std::string& filepath) {
    bool gz = has_gz_extension(filepath);
    znzFile fp = znzopen(filepath.c_str(), "rb", gz ? 1 : 0);
    if (znz_isnull(fp))
        throw std::runtime_error("Cannot open NIfTI file: " + filepath);

    nifti_1_header hdr;
    size_t nread = znzread(&hdr, sizeof(hdr), 1, fp);
    if (nread != 1) {
        znzclose(fp);
        throw std::runtime_error("Failed to read NIfTI header: " + filepath);
    }

    if (hdr.sizeof_hdr != 348) {
        znzclose(fp);
        throw std::runtime_error("Invalid NIfTI header (sizeof_hdr != 348): " + filepath);
    }

    long vox_offset = static_cast<long>(hdr.vox_offset);
    if (vox_offset < 352) vox_offset = 352;
    znzseek(fp, vox_offset, SEEK_SET);

    int nx = hdr.dim[1];
    int ny = hdr.dim[2];
    int nz = hdr.dim[3];
    size_t nvox = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

    int bpv = bytes_per_voxel(hdr.datatype);
    if (bpv == 0) {
        znzclose(fp);
        throw std::runtime_error("Unsupported datatype in: " + filepath);
    }

    std::vector<char> rawbuf(nvox * static_cast<size_t>(bpv));
    size_t got = znzread(rawbuf.data(), static_cast<size_t>(bpv), nvox, fp);
    znzclose(fp);

    if (got != nvox)
        throw std::runtime_error("Truncated voxel data in: " + filepath);

    NiftiVolume vol;
    vol.data = convert_to_float(rawbuf.data(), nvox, hdr.datatype, hdr.scl_slope, hdr.scl_inter);
    vol.nx = nx;
    vol.ny = ny;
    vol.nz = nz;
    vol.dx = std::fabs(hdr.pixdim[1]);
    vol.dy = std::fabs(hdr.pixdim[2]);
    vol.dz = std::fabs(hdr.pixdim[3]);

    std::copy(std::begin(hdr.srow_x), std::end(hdr.srow_x), vol.srow_x.begin());
    std::copy(std::begin(hdr.srow_y), std::end(hdr.srow_y), vol.srow_y.begin());
    std::copy(std::begin(hdr.srow_z), std::end(hdr.srow_z), vol.srow_z.begin());

    vol.qform_code = hdr.qform_code;
    vol.sform_code = hdr.sform_code;
    vol.quatern_b  = hdr.quatern_b;
    vol.quatern_c  = hdr.quatern_c;
    vol.quatern_d  = hdr.quatern_d;
    vol.qoffset_x  = hdr.qoffset_x;
    vol.qoffset_y  = hdr.qoffset_y;
    vol.qoffset_z  = hdr.qoffset_z;

    return vol;
}


void nifti::save(const std::string& filepath, const NiftiVolume& vol) {
    nifti_1_header hdr;
    std::memset(&hdr, 0, sizeof(hdr));

    hdr.sizeof_hdr = 348;
    hdr.dim[0] = 3;
    hdr.dim[1] = static_cast<int16_t>(vol.nx);
    hdr.dim[2] = static_cast<int16_t>(vol.ny);
    hdr.dim[3] = static_cast<int16_t>(vol.nz);
    hdr.dim[4] = 1;
    hdr.datatype = DT_FLOAT32;
    hdr.bitpix   = 32;
    hdr.pixdim[1] = vol.dx;
    hdr.pixdim[2] = vol.dy;
    hdr.pixdim[3] = vol.dz;
    hdr.vox_offset = 352.0f;
    hdr.scl_slope  = 1.0f;
    hdr.scl_inter  = 0.0f;
    hdr.xyzt_units = NIFTI_UNITS_MM;

    std::copy(vol.srow_x.begin(), vol.srow_x.end(), hdr.srow_x);
    std::copy(vol.srow_y.begin(), vol.srow_y.end(), hdr.srow_y);
    std::copy(vol.srow_z.begin(), vol.srow_z.end(), hdr.srow_z);
    hdr.qform_code = vol.qform_code;
    hdr.sform_code = vol.sform_code;
    hdr.quatern_b  = vol.quatern_b;
    hdr.quatern_c  = vol.quatern_c;
    hdr.quatern_d  = vol.quatern_d;
    hdr.qoffset_x  = vol.qoffset_x;
    hdr.qoffset_y  = vol.qoffset_y;
    hdr.qoffset_z  = vol.qoffset_z;

    std::memcpy(hdr.magic, NII_SINGLE_MAGIC, 4);

    bool gz = has_gz_extension(filepath);
    znzFile fp = znzopen(filepath.c_str(), "wb", gz ? 1 : 0);
    if (znz_isnull(fp))
        throw std::runtime_error("Cannot create output file: " + filepath);

    znzwrite(&hdr, sizeof(hdr), 1, fp);

    char ext[4] = {0, 0, 0, 0};
    znzwrite(ext, 1, 4, fp);

    znzwrite(vol.data.data(), sizeof(float), vol.nvoxels(), fp);
    znzclose(fp);
}


void nifti::save_labels(const std::string& filepath, const std::vector<int>& labels,
                         const NiftiVolume& reference)
{
    nifti_1_header hdr;
    std::memset(&hdr, 0, sizeof(hdr));

    hdr.sizeof_hdr = 348;
    hdr.dim[0] = 3;
    hdr.dim[1] = static_cast<int16_t>(reference.nx);
    hdr.dim[2] = static_cast<int16_t>(reference.ny);
    hdr.dim[3] = static_cast<int16_t>(reference.nz);
    hdr.dim[4] = 1;
    hdr.datatype  = DT_UINT8;
    hdr.bitpix    = 8;
    hdr.pixdim[1] = reference.dx;
    hdr.pixdim[2] = reference.dy;
    hdr.pixdim[3] = reference.dz;
    hdr.vox_offset = 352.0f;
    hdr.scl_slope  = 0.0f;
    hdr.scl_inter  = 0.0f;
    hdr.xyzt_units = NIFTI_UNITS_MM;
    hdr.intent_code = NIFTI_INTENT_LABEL;

    std::copy(reference.srow_x.begin(), reference.srow_x.end(), hdr.srow_x);
    std::copy(reference.srow_y.begin(), reference.srow_y.end(), hdr.srow_y);
    std::copy(reference.srow_z.begin(), reference.srow_z.end(), hdr.srow_z);
    hdr.qform_code = reference.qform_code;
    hdr.sform_code = reference.sform_code;
    hdr.quatern_b  = reference.quatern_b;
    hdr.quatern_c  = reference.quatern_c;
    hdr.quatern_d  = reference.quatern_d;
    hdr.qoffset_x  = reference.qoffset_x;
    hdr.qoffset_y  = reference.qoffset_y;
    hdr.qoffset_z  = reference.qoffset_z;

    std::memcpy(hdr.magic, NII_SINGLE_MAGIC, 4);

    bool gz = has_gz_extension(filepath);
    znzFile fp = znzopen(filepath.c_str(), "wb", gz ? 1 : 0);
    if (znz_isnull(fp))
        throw std::runtime_error("Cannot create output file: " + filepath);

    znzwrite(&hdr, sizeof(hdr), 1, fp);
    char ext[4] = {0, 0, 0, 0};
    znzwrite(ext, 1, 4, fp);

    std::vector<uint8_t> buf(labels.size());
    for (size_t i = 0; i < labels.size(); i++)
        buf[i] = static_cast<uint8_t>(std::clamp(labels[i], 0, 255));

    znzwrite(buf.data(), 1, buf.size(), fp);
    znzclose(fp);
}
