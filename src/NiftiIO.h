#pragma once

#include <string>
#include <vector>
#include <array>
#include <cstdint>

// 3D volume as flat float array (x fastest). Stores sform/qform for output.

struct NiftiVolume {
    std::vector<float> data;
    int nx = 0, ny = 0, nz = 0;           /* voxel dimensions */
    float dx = 1.0f, dy = 1.0f, dz = 1.0f; /* voxel spacing in mm */
    std::array<float, 4> srow_x = {1,0,0,0};
    std::array<float, 4> srow_y = {0,1,0,0};
    std::array<float, 4> srow_z = {0,0,1,0};
    int16_t qform_code = 0;
    int16_t sform_code = 0;
    float quatern_b = 0, quatern_c = 0, quatern_d = 0;
    float qoffset_x = 0, qoffset_y = 0, qoffset_z = 0;

    inline size_t nvoxels() const { return (size_t)nx * ny * nz; }
    inline float& at(int x, int y, int z) {
        return data[static_cast<size_t>(z) * static_cast<size_t>(ny) * static_cast<size_t>(nx)
                   + static_cast<size_t>(y) * static_cast<size_t>(nx)
                   + static_cast<size_t>(x)];
    }
    inline float at(int x, int y, int z) const {
        return data[static_cast<size_t>(z) * static_cast<size_t>(ny) * static_cast<size_t>(nx)
                   + static_cast<size_t>(y) * static_cast<size_t>(nx)
                   + static_cast<size_t>(x)];
    }
};

namespace nifti {

/* Read a .nii or .nii.gz file, converting any datatype to float. */
NiftiVolume load(const std::string& filepath);

/* Write a float volume as a NIfTI file. Uses .nii.gz if path ends in .gz. */
void save(const std::string& filepath, const NiftiVolume& vol);

/* Write an integer label map (e.g. segmentation mask) preserving vol's spatial info. */
void save_labels(const std::string& filepath, const std::vector<int>& labels, const NiftiVolume& reference);

} // namespace nifti
