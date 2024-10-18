#include <H5Cpp.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

class VelocityReader {
 public:
  VelocityReader(const std::string& filename) : filename_(filename) {
    try {
      file_ = H5::H5File(filename, H5F_ACC_RDONLY);
      time_dataset_ = file_.openDataSet("time");
      velocities_dataset_ = file_.openDataSet("velocities");

      H5::DataSpace time_dataspace = time_dataset_.getSpace();
      H5::DataSpace velocities_dataspace = velocities_dataset_.getSpace();

      hsize_t dims[3];
      int ndims = velocities_dataspace.getSimpleExtentDims(dims, NULL);

      num_frames_ = dims[0];
      num_atoms_ = dims[1];
    } catch (H5::Exception& e) {
      std::cerr << "Error opening file: " << e.getDetailMsg() << std::endl;
      throw;
    }
  }

  std::vector<float> readTimeSteps(hsize_t start, hsize_t count) {
    std::vector<float> time_steps(count);
    hsize_t dim[1] = {count};
    H5::DataSpace memspace(1, dim);
    H5::DataSpace filespace = time_dataset_.getSpace();
    hsize_t offset[1] = {start};
    hsize_t count_arr[1] = {count};
    filespace.selectHyperslab(H5S_SELECT_SET, count_arr, offset);
    time_dataset_.read(time_steps.data(), H5::PredType::NATIVE_FLOAT, memspace,
                       filespace);
    return time_steps;
  }

  std::vector<float> readVelocities(hsize_t frame) {
    std::vector<float> velocities(num_atoms_ * 3);
    hsize_t count[3] = {1, num_atoms_, 3};
    hsize_t offset[3] = {frame, 0, 0};
    H5::DataSpace memspace(3, count);
    H5::DataSpace filespace = velocities_dataset_.getSpace();
    filespace.selectHyperslab(H5S_SELECT_SET, count, offset);
    velocities_dataset_.read(velocities.data(), H5::PredType::NATIVE_FLOAT,
                             memspace, filespace);
    return velocities;
  }

  hsize_t getNumFrames() const { return num_frames_; }
  hsize_t getNumAtoms() const { return num_atoms_; }

 private:
  std::string filename_;
  H5::H5File file_;
  H5::DataSet time_dataset_;
  H5::DataSet velocities_dataset_;
  hsize_t num_frames_;
  hsize_t num_atoms_;
};

class VelocityAdjuster {
 public:
  std::vector<float> adjustVelocities(const std::vector<float>& prev_vel,
                                      const std::vector<float>& curr_vel) {
    std::vector<float> adjusted_vel(prev_vel.size());
#pragma omp parallel for
    for (size_t i = 0; i < prev_vel.size(); ++i) {
      adjusted_vel[i] = 0.5f * (prev_vel[i] + curr_vel[i]);
    }
    return adjusted_vel;
  }
};

class VelocityWriter {
 public:
  VelocityWriter(const std::string& filename, hsize_t num_atoms)
      : filename_(filename), num_atoms_(num_atoms) {
    try {
      H5::H5File file(filename, H5F_ACC_TRUNC);

      // Create extensible dataspace for time
      hsize_t time_dims[1] = {0};
      hsize_t time_maxdims[1] = {H5S_UNLIMITED};
      H5::DataSpace time_dataspace(1, time_dims, time_maxdims);

      // Create property list for chunked dataset
      H5::DSetCreatPropList time_plist;
      hsize_t time_chunk_dims[1] = {1000};
      time_plist.setChunk(1, time_chunk_dims);

      // Create the time dataset
      time_dataset_ = file.createDataSet("time", H5::PredType::NATIVE_FLOAT,
                                         time_dataspace, time_plist);

      // Create extensible dataspace for velocities
      hsize_t vel_dims[3] = {0, num_atoms, 3};
      hsize_t vel_maxdims[3] = {H5S_UNLIMITED, num_atoms, 3};
      H5::DataSpace vel_dataspace(3, vel_dims, vel_maxdims);

      // Create property list for chunked dataset
      H5::DSetCreatPropList vel_plist;
      hsize_t vel_chunk_dims[3] = {100, num_atoms, 3};
      vel_plist.setChunk(3, vel_chunk_dims);

      // Create the velocities dataset
      velocities_dataset_ = file.createDataSet(
          "velocities", H5::PredType::NATIVE_FLOAT, vel_dataspace, vel_plist);

      // Add attributes
      H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
      H5::Attribute time_units_attr = time_dataset_.createAttribute(
          "units", str_type, H5::DataSpace(H5S_SCALAR));
      time_units_attr.write(str_type, std::string("picosecond"));

      H5::Attribute vel_units_attr = velocities_dataset_.createAttribute(
          "units", str_type, H5::DataSpace(H5S_SCALAR));
      vel_units_attr.write(str_type, std::string("angstrom/picosecond"));

      H5::Attribute scale_factor_attr = velocities_dataset_.createAttribute(
          "scale_factor", H5::PredType::NATIVE_FLOAT,
          H5::DataSpace(H5S_SCALAR));
      float scale_factor = 20.455f;
      scale_factor_attr.write(H5::PredType::NATIVE_FLOAT, &scale_factor);

    } catch (H5::Exception& e) {
      std::cerr << "Error creating output file: " << e.getDetailMsg()
                << std::endl;
      throw;
    }
  }

  void writeTimeSteps(const std::vector<float>& time_steps,
                      hsize_t start_frame) {
    try {
      hsize_t dims[1] = {time_steps.size()};
      hsize_t offset[1] = {start_frame};
      H5::DataSpace memspace(1, dims);

      // Extend the dataset
      hsize_t new_size[1] = {start_frame + time_steps.size()};
      time_dataset_.extend(new_size);

      // Get the new file dataspace
      H5::DataSpace filespace = time_dataset_.getSpace();
      filespace.selectHyperslab(H5S_SELECT_SET, dims, offset);

      time_dataset_.write(time_steps.data(), H5::PredType::NATIVE_FLOAT,
                          memspace, filespace);
    } catch (H5::Exception& e) {
      std::cerr << "Error writing time steps: " << e.getDetailMsg()
                << std::endl;
      throw;
    }
  }

  void writeVelocities(const std::vector<std::vector<float>>& velocities,
                       hsize_t start_frame) {
    try {
      hsize_t dims[3] = {velocities.size(), num_atoms_, 3};
      hsize_t offset[3] = {start_frame, 0, 0};
      H5::DataSpace memspace(3, dims);

      // Extend the dataset
      hsize_t new_size[3] = {start_frame + velocities.size(), num_atoms_, 3};
      velocities_dataset_.extend(new_size);

      // Get the new file dataspace
      H5::DataSpace filespace = velocities_dataset_.getSpace();
      filespace.selectHyperslab(H5S_SELECT_SET, dims, offset);

      std::vector<float> flattened_velocities;
      for (const auto& frame : velocities) {
        flattened_velocities.insert(flattened_velocities.end(), frame.begin(),
                                    frame.end());
      }

      velocities_dataset_.write(flattened_velocities.data(),
                                H5::PredType::NATIVE_FLOAT, memspace,
                                filespace);
    } catch (H5::Exception& e) {
      std::cerr << "Error writing velocities: " << e.getDetailMsg()
                << std::endl;
      throw;
    }
  }

 private:
  std::string filename_;
  hsize_t num_atoms_;
  H5::DataSet time_dataset_;
  H5::DataSet velocities_dataset_;
};

void adjustVelocities(const std::string& input_file,
                      const std::string& output_file,
                      const std::array<int, 3>& input_steps,
                      const std::array<int, 3>& output_steps) {
  VelocityReader reader(input_file);
  hsize_t num_atoms = reader.getNumAtoms();
  VelocityWriter writer(output_file, num_atoms);
  VelocityAdjuster adjuster;

  hsize_t num_frames = reader.getNumFrames();
  int step_begin = input_steps[0];
  int step_end =
      (input_steps[1] == -1)
          ? num_frames
          : std::min(static_cast<hsize_t>(input_steps[1]), num_frames);
  int step_interval = input_steps[2];

  std::vector<float> prev_velocities;
  std::vector<std::vector<float>> adjusted_velocities;
  std::vector<float> adjusted_time_steps;

#pragma omp parallel
  {
    std::vector<float> local_prev_velocities;
    std::vector<std::vector<float>> local_adjusted_velocities;
    std::vector<float> local_adjusted_time_steps;

#pragma omp for schedule(dynamic)
    for (int i = step_begin; i < step_end; i += step_interval) {
      auto curr_velocities = reader.readVelocities(i);

      if (!local_prev_velocities.empty()) {
        auto adjusted =
            adjuster.adjustVelocities(local_prev_velocities, curr_velocities);
        local_adjusted_velocities.push_back(adjusted);
        local_adjusted_time_steps.push_back(static_cast<float>(i - 1) *
                                            0.01f);  // Assuming dt = 0.01
      }

      local_prev_velocities = curr_velocities;

      if (local_adjusted_velocities.size() == 10000 || i == step_end - 1) {
#pragma omp critical
        {
          adjusted_velocities.insert(adjusted_velocities.end(),
                                     local_adjusted_velocities.begin(),
                                     local_adjusted_velocities.end());
          adjusted_time_steps.insert(adjusted_time_steps.end(),
                                     local_adjusted_time_steps.begin(),
                                     local_adjusted_time_steps.end());
        }
        local_adjusted_velocities.clear();
        local_adjusted_time_steps.clear();
      }
    }
  }

  // Write the accumulated data outside the parallel region
  hsize_t write_offset = 0;
  while (!adjusted_velocities.empty()) {
    hsize_t chunk_size =
        std::min(adjusted_velocities.size(), static_cast<size_t>(10000));
    std::vector<std::vector<float>> chunk_velocities(
        adjusted_velocities.begin(), adjusted_velocities.begin() + chunk_size);
    std::vector<float> chunk_time_steps(
        adjusted_time_steps.begin(), adjusted_time_steps.begin() + chunk_size);

    writer.writeTimeSteps(chunk_time_steps, write_offset);
    writer.writeVelocities(chunk_velocities, write_offset);

    adjusted_velocities.erase(adjusted_velocities.begin(),
                              adjusted_velocities.begin() + chunk_size);
    adjusted_time_steps.erase(adjusted_time_steps.begin(),
                              adjusted_time_steps.begin() + chunk_size);
    write_offset += chunk_size;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 9) {
    std::cerr
        << "Usage: " << argv[0]
        << " <input_file> <output_file> <input_begin> <input_end> "
           "<input_interval> <output_begin> <output_end> <output_interval>"
        << std::endl;
    return 1;
  }

  std::string input_file = argv[1];
  std::string output_file = argv[2];
  std::array<int, 3> input_steps = {std::stoi(argv[3]), std::stoi(argv[4]),
                                    std::stoi(argv[5])};
  std::array<int, 3> output_steps = {std::stoi(argv[6]), std::stoi(argv[7]),
                                     std::stoi(argv[8])};

  std::cout << "Input file: " << input_file << std::endl;
  std::cout << "Output file: " << output_file << std::endl;
  // store elapsed time
  double start_time = omp_get_wtime();
  try {
    adjustVelocities(input_file, output_file, input_steps, output_steps);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  // print elapsed time
  double elapsed_time = omp_get_wtime() - start_time;
  std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;

  return 0;
}