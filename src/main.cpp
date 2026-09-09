#include <iostream>
#include <string>

#include "pipeline.hpp"

int main(int argc, char** argv)
{
    if (argc != 3 || std::string(argv[1]) != "--config")
    {
        std::cerr << "usage: lio_visual_ba_pipeline --config PIPELINE.yaml\n";
        return 2;
    }
    auto config = lio_visual_ba::Pipeline::LoadPipelineConfig(argv[2]);
    if (!config.ok())
    {
        std::cerr << config.status().ToString() << '\n';
        return 1;
    }
    auto result = lio_visual_ba::Pipeline::RunPipeline(config.value());
    if (!result.ok())
    {
        std::cerr << result.status().ToString() << '\n';
        return 1;
    }
    std::cout << "C++ sparse reconstruction complete\n"
              << "  cameras: " << result.value().backend.reconstruction.cameras.size() << '\n'
              << "  landmarks: " << result.value().backend.reconstruction.landmarks.size() << '\n'
              << "  verified matches: " << result.value().verified_matches << '\n'
              << "  output: " << config.value().output_directory << '\n';
    return 0;
}
