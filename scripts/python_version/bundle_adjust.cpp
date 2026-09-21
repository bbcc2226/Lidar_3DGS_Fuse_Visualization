#include <algorithm>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

using Json = nlohmann::json;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Quat = Eigen::Quaterniond;

struct CameraState {
    double ts;
    double q[4];
    double t[3];
    double pq[4];
    double pt[3];
    std::string name;
};

struct Point3D {
    double x[3];
    double initial_x[3];
    int r;
    int g;
    int b;
};

struct Observation {
    int p;
    int f;
    double u;
    double v;
};

struct HeldOutMatch {
    int a;
    int b;
    double u;
    double v;
    double x;
    double y;
};

Mat3 read_intrinsics(const fs::path& path)
{
    std::ifstream file(path);
    Mat3 K;
    std::string line;

    for (int row = 0; row < 3; ++row) {
        std::getline(file, line);
        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream ss(line);
        for (int col = 0; col < 3; ++col) {
            ss >> K(row, col);
        }
    }

    return K;
}

std::vector<CameraState> read_camera_states(const fs::path& pose_path, const fs::path& stamps_path)
{
    std::vector<std::string> names;
    std::ifstream stamps_file(stamps_path);
    std::string name;
    double ts = 0.0;
    while (stamps_file >> name >> ts) {
        names.push_back(name);
    }

    std::ifstream pose_file(pose_path);
    std::vector<CameraState> cameras;
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double qw = 0.0;
    int index = 0;

    while (pose_file >> ts >> x >> y >> w >> qx >> qy >> qz >> qw) {
        CameraState camera{};
        camera.ts = ts;
        camera.q[0] = qw;
        camera.q[1] = qx;
        camera.q[2] = qy;
        camera.q[3] = qz;
        camera.t[0] = x;
        camera.t[1] = y;
        camera.t[2] = w;
        camera.pq[0] = qw;
        camera.pq[1] = qx;
        camera.pq[2] = qy;
        camera.pq[3] = qz;
        camera.pt[0] = x;
        camera.pt[1] = y;
        camera.pt[2] = w;
        camera.name = names.at(index++);
        cameras.push_back(camera);
    }

    return cameras;
}

void initialize_camera_states(const fs::path& pose_path, std::vector<CameraState>& cameras)
{
    std::ifstream pose_file(pose_path);
    double ts, x, y, z, qx, qy, qz, qw;
    size_t index = 0;
    while (index < cameras.size() && pose_file >> ts >> x >> y >> z >> qx >> qy >> qz >> qw) {
        cameras[index].q[0] = qw; cameras[index].q[1] = qx; cameras[index].q[2] = qy; cameras[index].q[3] = qz;
        cameras[index].t[0] = x; cameras[index].t[1] = y; cameras[index].t[2] = z;
        ++index;
    }
    if (index != cameras.size()) throw std::runtime_error("initial pose count does not match cameras");
}

std::vector<Point3D> read_point_cloud(const fs::path& path)
{
    std::ifstream file(path);
    std::string line;
    int count = 0;

    while (std::getline(file, line)) {
        if (line.rfind("element vertex ", 0) == 0) {
            count = std::stoi(line.substr(15));
        }
        if (line == "end_header") {
            break;
        }
    }

    std::vector<Point3D> points(count);
    for (auto& point : points) {
        do { std::getline(file, line); } while (line.empty() && file);
        std::stringstream vertex(line);
        vertex >> point.x[0] >> point.x[1] >> point.x[2] >> point.r >> point.g >> point.b;
        if (!vertex) throw std::runtime_error("invalid PLY vertex record");
        std::copy(point.x, point.x + 3, point.initial_x);
    }

    return points;
}

std::vector<Observation> read_observations(const fs::path& path)
{
    std::ifstream file(path);
    std::string header;
    std::getline(file, header);

    std::vector<Observation> observations;
    Observation obs{};
    char separator = 0;

    while (file >> obs.p >> separator >> obs.f >> separator >> obs.u >> separator >> obs.v) {
        observations.push_back(obs);
    }

    return observations;
}

std::vector<HeldOutMatch> read_heldout_matches(const fs::path& path)
{
    std::ifstream file(path);
    std::string header;
    std::getline(file, header);

    std::vector<HeldOutMatch> matches;
    HeldOutMatch match{};
    char separator = 0;

    while (file >> match.a >> separator >> match.b >> separator >> match.u >> separator >> match.v >> separator >> match.x >> separator >> match.y) {
        matches.push_back(match);
    }

    return matches;
}

struct ReprojectionResidual {
    double u;
    double v;
    double fx;
    double fy;
    double cx;
    double cy;
    double weight;

    template <typename T>
    bool operator()(const T* q, const T* t, const T* x, T* residuals) const
    {
        T point_in_camera[3] = {
            x[0] - t[0],
            x[1] - t[1],
            x[2] - t[2],
        };
        T conjugate_q[4] = {q[0], -q[1], -q[2], -q[3]};
        T rotated[3];

        ceres::QuaternionRotatePoint(conjugate_q, point_in_camera, rotated);

        residuals[0] = T(weight) * (T(fx) * rotated[0] / rotated[2] + T(cx) - T(u));
        residuals[1] = T(weight) * (T(fy) * rotated[1] / rotated[2] + T(cy) - T(v));
        return true;
    }
};

struct IntrinsicReprojectionResidual {
    double u, v;
    template <typename T>
    bool operator()(const T* q, const T* t, const T* x, const T* k, T* residuals) const {
        T point[3]={x[0]-t[0],x[1]-t[1],x[2]-t[2]};
        T qc[4]={q[0],-q[1],-q[2],-q[3]}, rotated[3];
        ceres::QuaternionRotatePoint(qc,point,rotated);
        residuals[0]=k[0]*rotated[0]/rotated[2]+k[2]-T(u);
        residuals[1]=k[1]*rotated[1]/rotated[2]+k[3]-T(v);
        return true;
    }
};

struct IntrinsicPriorResidual {
    double fx,fy,cx,cy;
    template <typename T> bool operator()(const T* k,T* r) const {
        r[0]=(k[0]-T(fx))/T(6.0); r[1]=(k[1]-T(fy))/T(6.0);
        r[2]=(k[2]-T(cx))/T(3.0); r[3]=(k[3]-T(cy))/T(3.0); return true;
    }
};

// Residual weights are 1/sigma: translation in 1/m, rotation in 1/rad. Set from CLI.
double g_translation_prior_weight = 2.0;
double g_rotation_prior_weight = 10.0;
// Huber(1.0) turns linear beyond 1/weight, so a robust prior cannot be made strong.
bool g_robust_pose_prior = true;

ceres::LossFunction* pose_prior_loss()
{
    return g_robust_pose_prior ? new ceres::HuberLoss(1.0) : nullptr;
}

struct TranslationPriorResidual {
    double x;
    double y;
    double z;
    double weight;

    template <typename T>
    bool operator()(const T* t, T* residuals) const
    {
        residuals[0] = T(weight) * (t[0] - T(x));
        residuals[1] = T(weight) * (t[1] - T(y));
        residuals[2] = T(weight) * (t[2] - T(z));
        return true;
    }
};

struct LandmarkPriorResidual {
    double x;
    double y;
    double z;
    double weight;

    template <typename T>
    bool operator()(const T* point, T* residuals) const
    {
        residuals[0] = T(weight) * (point[0] - T(x));
        residuals[1] = T(weight) * (point[1] - T(y));
        residuals[2] = T(weight) * (point[2] - T(z));
        return true;
    }
};

struct RotationPriorResidual {
    double a;
    double b;
    double c;
    double d;
    double weight;

    template <typename T>
    bool operator()(const T* q, T* residuals) const
    {
        T prior_q[4] = {T(a), T(-b), T(-c), T(-d)};
        T product[4];
        ceres::QuaternionProduct(prior_q, q, product);

        for (int i = 0; i < 3; ++i) {
            residuals[i] = T(2.0 * weight) * product[i + 1];
        }
        return true;
    }
};

void run_local_bundle_adjustment(
    std::vector<CameraState>& cameras,
    std::vector<Point3D>& points,
    const std::vector<Observation>& observations,
    const Mat3& K,
    int start_index,
    int end_index,
    int num_iterations,
    const char* tag)
{
    ceres::Problem problem;
    const bool is_global = start_index == 0 && end_index == static_cast<int>(cameras.size());
    std::vector<char> touched(points.size(), false);
    std::vector<std::vector<int>> point_frames(points.size());
    for (const auto& obs : observations) {
        if (obs.p < 0 || obs.p >= static_cast<int>(points.size()) ||
            obs.f < 0 || obs.f >= static_cast<int>(cameras.size())) continue;
        point_frames[obs.p].push_back(obs.f);
        if (obs.f >= start_index && obs.f < end_index) touched[obs.p] = true;
    }
    std::vector<double> point_weights(points.size(), 1.0);
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& frames = point_frames[i];
        if (frames.size() < 2) continue;
        const Vec3 landmark(points[i].initial_x);
        double minimum_dot = 1.0;
        for (size_t a = 0; a < frames.size(); ++a) {
            Vec3 ray_a = landmark - Vec3(cameras[frames[a]].t);
            if (ray_a.norm() < 1e-9) continue;
            ray_a.normalize();
            for (size_t b = a + 1; b < frames.size(); ++b) {
                Vec3 ray_b = landmark - Vec3(cameras[frames[b]].t);
                if (ray_b.norm() < 1e-9) continue;
                ray_b.normalize();
                minimum_dot = std::min(minimum_dot, std::clamp(ray_a.dot(ray_b), -1.0, 1.0));
            }
        }
        const double parallax = std::acos(std::clamp(minimum_dot, -1.0, 1.0));
        const double angle_factor = std::clamp(std::sin(parallax) / std::sin(2.0 * M_PI / 180.0), 0.75, 1.15);
        const double track_factor = std::clamp(std::sqrt(double(frames.size()) / 3.0), 1.0, 1.20);
        point_weights[i] = 1.0;
    }

    for (int i = start_index; i < end_index; ++i) {
        problem.AddParameterBlock(cameras[i].q, 4, new ceres::QuaternionManifold);
        problem.AddParameterBlock(cameras[i].t, 3);

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TranslationPriorResidual, 3, 3>(
                new TranslationPriorResidual{cameras[i].pt[0], cameras[i].pt[1], cameras[i].pt[2], g_translation_prior_weight}),
            pose_prior_loss(),
            cameras[i].t);

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RotationPriorResidual, 3, 4>(
                new RotationPriorResidual{cameras[i].pq[0], cameras[i].pq[1], cameras[i].pq[2], cameras[i].pq[3], g_rotation_prior_weight}),
            pose_prior_loss(),
            cameras[i].q);
    }

    if (!is_global) {
        for (int i = 0; i < static_cast<int>(cameras.size()); ++i) {
            if (i >= start_index && i < end_index) continue;
            problem.AddParameterBlock(cameras[i].q, 4, new ceres::QuaternionManifold);
            problem.AddParameterBlock(cameras[i].t, 3);
            problem.SetParameterBlockConstant(cameras[i].q);
            problem.SetParameterBlockConstant(cameras[i].t);
        }
    }

    std::vector<int> used(points.size(), 0);
    for (const auto& obs : observations) {
        if (obs.p >= 0 && obs.p < static_cast<int>(points.size()) && touched[obs.p]) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 4, 3, 3>(
                    new ReprojectionResidual{obs.u, obs.v, K(0, 0), K(1, 1), K(0, 2), K(1, 2), point_weights[obs.p]}),
                new ceres::CauchyLoss(2.0),
                cameras[obs.f].q,
                cameras[obs.f].t,
                points[obs.p].x);
            used[obs.p]++;
        }
    }

    for (size_t i = 0; i < points.size(); ++i) {
        if (!used[i]) {
            problem.AddParameterBlock(points[i].x, 3);
            problem.SetParameterBlockConstant(points[i].x);
        }
        if (used[i]) {
            double minimum_dot = 1.0;
            const Vec3 landmark(points[i].initial_x);
            const auto& frames = point_frames[i];
            for (size_t a = 0; a < frames.size(); ++a) {
                Vec3 ray_a = landmark - Vec3(cameras[frames[a]].t);
                if (ray_a.norm() < 1e-9) continue;
                ray_a.normalize();
                for (size_t b = a + 1; b < frames.size(); ++b) {
                    Vec3 ray_b = landmark - Vec3(cameras[frames[b]].t);
                    if (ray_b.norm() < 1e-9) continue;
                    ray_b.normalize();
                    minimum_dot = std::min(minimum_dot, std::clamp(ray_a.dot(ray_b), -1.0, 1.0));
                }
            }
            const double max_parallax = std::acos(std::clamp(minimum_dot, -1.0, 1.0));
            const double track_factor = std::min(2.0, std::sqrt(double(std::max<size_t>(3, frames.size())) / 3.0));
            const double sigma = std::clamp((0.10 + 2.0 * std::sin(max_parallax)) * track_factor, 0.10, 1.0);
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<LandmarkPriorResidual, 3, 3>(
                    new LandmarkPriorResidual{points[i].initial_x[0], points[i].initial_x[1], points[i].initial_x[2], 1.0 / sigma}),
                new ceres::HuberLoss(1.0), points[i].x);
        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = num_iterations;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.num_threads = std::max(1u, std::thread::hardware_concurrency());

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << tag << " " << summary.BriefReport() << '\n';
}

void run_intrinsic_bundle_adjustment(std::vector<CameraState>& cameras,
    std::vector<Point3D>& points,const std::vector<Observation>& observations,
    double k[4],const double prior[4]) {
    ceres::Problem problem; problem.AddParameterBlock(k,4);
    problem.SetParameterLowerBound(k,0,prior[0]*0.99); problem.SetParameterUpperBound(k,0,prior[0]*1.01);
    problem.SetParameterLowerBound(k,1,prior[1]*0.99); problem.SetParameterUpperBound(k,1,prior[1]*1.01);
    problem.SetParameterLowerBound(k,2,prior[2]-5.0); problem.SetParameterUpperBound(k,2,prior[2]+5.0);
    problem.SetParameterLowerBound(k,3,prior[3]-5.0); problem.SetParameterUpperBound(k,3,prior[3]+5.0);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<IntrinsicPriorResidual,4,4>(
        new IntrinsicPriorResidual{prior[0],prior[1],prior[2],prior[3]}),nullptr,k);
    for(auto& c:cameras) {
        problem.AddParameterBlock(c.q,4,new ceres::QuaternionManifold); problem.AddParameterBlock(c.t,3);
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<TranslationPriorResidual,3,3>(
            new TranslationPriorResidual{c.pt[0],c.pt[1],c.pt[2],g_translation_prior_weight}),pose_prior_loss(),c.t);
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<RotationPriorResidual,3,4>(
            new RotationPriorResidual{c.pq[0],c.pq[1],c.pq[2],c.pq[3],g_rotation_prior_weight}),pose_prior_loss(),c.q);
    }
    std::vector<char> used(points.size(),false);
    for(const auto& o:observations) if(o.p>=0&&o.p<(int)points.size()&&o.f>=0&&o.f<(int)cameras.size()) {
        used[o.p]=true; problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<IntrinsicReprojectionResidual,2,4,3,3,4>(new IntrinsicReprojectionResidual{o.u,o.v}),
            new ceres::CauchyLoss(2.0),cameras[o.f].q,cameras[o.f].t,points[o.p].x,k);
    }
    for(size_t i=0;i<points.size();++i) {
        problem.AddParameterBlock(points[i].x,3);
        if(!used[i]) problem.SetParameterBlockConstant(points[i].x);
        else problem.AddResidualBlock(new ceres::AutoDiffCostFunction<LandmarkPriorResidual,3,3>(
            new LandmarkPriorResidual{points[i].initial_x[0],points[i].initial_x[1],points[i].initial_x[2],1.0}),
            new ceres::HuberLoss(1.0),points[i].x);
    }
    ceres::Solver::Options options; options.max_num_iterations=50; options.linear_solver_type=ceres::SPARSE_SCHUR;
    options.num_threads=std::max(1u,std::thread::hardware_concurrency()); ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary); std::cout<<"INTRINSIC "<<summary.BriefReport()<<'\n';
}

Vec3 transform_to_camera_frame(const CameraState& camera, const Vec3& point_world)
{
    Quat q(camera.q[0], camera.q[1], camera.q[2], camera.q[3]);
    return q.conjugate() * (point_world - Vec3(camera.t));
}

double reprojection_error(const CameraState& camera, const Point3D& point, const Observation& obs, const Mat3& K)
{
    Vec3 p_camera = transform_to_camera_frame(camera, Vec3(point.x));
    if (p_camera.z() <= 0.0) {
        return 1e6;
    }

    const double u = K(0, 0) * p_camera.x() / p_camera.z() + K(0, 2);
    const double v = K(1, 1) * p_camera.y() / p_camera.z() + K(1, 2);
    return std::hypot(u - obs.u, v - obs.v);
}

std::vector<Observation> prune_observations(
    const std::vector<CameraState>& cameras,
    const std::vector<Point3D>& points,
    const std::vector<Observation>& observations,
    const Mat3& K, double threshold)
{
    std::vector<int> counts(points.size(), 0);
    std::vector<double> errors(observations.size(), 1e6);
    for (size_t i = 0; i < observations.size(); ++i) {
        const auto& obs = observations[i];
        errors[i] = reprojection_error(cameras[obs.f], points[obs.p], obs, K);
        if (errors[i] <= threshold) counts[obs.p]++;
    }
    std::vector<Observation> output;
    output.reserve(observations.size());
    for (size_t i = 0; i < observations.size(); ++i)
        if (errors[i] <= threshold && counts[observations[i].p] >= 3) output.push_back(observations[i]);
    return output;
}

void write_observations(const fs::path& path, const std::vector<Observation>& observations)
{
    std::ofstream file(path); file << "track,frame,u,v\n";
    for (const auto& obs : observations) file << obs.p << ',' << obs.f << ',' << obs.u << ',' << obs.v << '\n';
}

Mat3 skew_matrix(const Vec3& v)
{
    Mat3 S;
    S << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
    return S;
}

double epipolar_error(const CameraState& camera_a, const CameraState& camera_b, const HeldOutMatch& match, const Mat3& K)
{
    const Quat qa(camera_a.q[0], camera_a.q[1], camera_a.q[2], camera_a.q[3]);
    const Quat qb(camera_b.q[0], camera_b.q[1], camera_b.q[2], camera_b.q[3]);

    Mat3 rotation = qb.conjugate().toRotationMatrix() * qa.toRotationMatrix();
    Vec3 translation = qb.conjugate() * (Vec3(camera_a.t) - Vec3(camera_b.t));
    Mat3 fundamental = K.inverse().transpose() * skew_matrix(translation) * rotation * K.inverse();

    const Vec3 x(match.u, match.v, 1.0);
    const Vec3 y(match.x, match.y, 1.0);
    const Vec3 line_from_x = fundamental * x;
    const Vec3 line_from_y = fundamental.transpose() * y;

    const double numerator = std::abs(y.dot(line_from_x));
    const double denominator = std::sqrt(std::max(1e-12, line_from_x.head<2>().squaredNorm() + line_from_y.head<2>().squaredNorm()));
    return numerator / denominator;
}

double quantile(const std::vector<double>& values, double q)
{
    if (values.empty()) {
        return 1e9;
    }

    const size_t index = std::min(values.size() - 1, static_cast<size_t>(q * (values.size() - 1)));
    std::vector<double> sorted = values;
    std::nth_element(sorted.begin(), sorted.begin() + index, sorted.end());
    return sorted[index];
}

Json select_landmarks_for_ba(
    std::vector<Point3D>& points, std::vector<Observation>& observations,
    const std::vector<CameraState>& cameras, const Mat3& K,
    int max_landmarks, int min_landmarks_per_image)
{
    const size_t input_points = points.size();
    const size_t input_observations = observations.size();
    Json stats = {{"enabled", max_landmarks > 0}, {"input_landmarks", input_points},
        {"input_observations", input_observations}, {"max_landmarks", max_landmarks},
        {"minimum_landmarks_per_image", min_landmarks_per_image}};
    if (max_landmarks <= 0 || points.size() <= static_cast<size_t>(max_landmarks)) {
        std::vector<char> covered(cameras.size(), false);
        for (const auto& obs : observations)
            if (obs.f >= 0 && obs.f < static_cast<int>(covered.size())) covered[obs.f] = true;
        stats["selected_landmarks"] = input_points;
        stats["selected_observations"] = input_observations;
        stats["coverable_images"] = std::count(covered.begin(), covered.end(), true);
        stats["covered_images"] = stats["coverable_images"];
        return stats;
    }

    std::vector<std::vector<int>> point_observations(points.size());
    for (size_t i = 0; i < observations.size(); ++i) {
        const auto& obs = observations[i];
        if (obs.p >= 0 && obs.p < static_cast<int>(points.size()) &&
            obs.f >= 0 && obs.f < static_cast<int>(cameras.size()))
            point_observations[obs.p].push_back(static_cast<int>(i));
    }
    struct Candidate { int point; double quality; };
    std::vector<Candidate> ranked;
    std::vector<std::vector<Candidate>> by_image(cameras.size());
    for (size_t p = 0; p < points.size(); ++p) {
        const auto& indices = point_observations[p];
        if (indices.size() < 3) continue;
        int first_frame = static_cast<int>(cameras.size()), last_frame = -1;
        double minimum_dot = 1.0;
        std::vector<double> errors;
        std::vector<Vec3> rays;
        errors.reserve(indices.size()); rays.reserve(indices.size());
        for (int index : indices) {
            const auto& obs = observations[index];
            first_frame = std::min(first_frame, obs.f);
            last_frame = std::max(last_frame, obs.f);
            errors.push_back(reprojection_error(cameras[obs.f], points[p], obs, K));
            Vec3 ray = Vec3(points[p].x) - Vec3(cameras[obs.f].t);
            if (ray.norm() > 1e-9) rays.push_back(ray.normalized());
        }
        for (size_t a = 0; a < rays.size(); ++a)
            for (size_t b = a + 1; b < rays.size(); ++b)
                minimum_dot = std::min(minimum_dot,
                    std::clamp(rays[a].dot(rays[b]), -1.0, 1.0));
        const double parallax = std::acos(std::clamp(minimum_dot, -1.0, 1.0));
        const double median_error = quantile(errors, 0.5);
        if (!std::isfinite(median_error)) continue;
        const double quality = 2.0 * std::log1p(static_cast<double>(indices.size()))
            + 0.75 * std::log1p(static_cast<double>(last_frame - first_frame))
            + 2.0 * std::sin(parallax) - std::log1p(median_error);
        const Candidate candidate{static_cast<int>(p), quality};
        ranked.push_back(candidate);
        for (int index : indices) by_image[observations[index].f].push_back(candidate);
    }
    const auto better = [](const Candidate& a, const Candidate& b) {
        return a.quality != b.quality ? a.quality > b.quality : a.point < b.point;
    };
    std::sort(ranked.begin(), ranked.end(), better);
    for (auto& candidates : by_image) std::sort(candidates.begin(), candidates.end(), better);

    std::vector<char> selected(points.size(), false);
    std::vector<int> image_counts(cameras.size(), 0);
    std::vector<char> coverable(cameras.size(), false);
    for (size_t f = 0; f < cameras.size(); ++f) coverable[f] = !by_image[f].empty();
    int selected_count = 0;
    const auto add_point = [&](int p) {
        if (selected[p] || selected_count >= max_landmarks) return false;
        selected[p] = true; ++selected_count;
        for (int index : point_observations[p]) ++image_counts[observations[index].f];
        return true;
    };
    if (min_landmarks_per_image > 0) {
        std::vector<size_t> cursor(cameras.size(), 0);
        while (selected_count < max_landmarks) {
            int weakest = -1;
            for (size_t f = 0; f < cameras.size(); ++f) {
                while (cursor[f] < by_image[f].size() && selected[by_image[f][cursor[f]].point]) ++cursor[f];
                if (image_counts[f] < min_landmarks_per_image && cursor[f] < by_image[f].size() &&
                    (weakest < 0 || image_counts[f] < image_counts[weakest])) weakest = static_cast<int>(f);
            }
            if (weakest < 0) break;
            add_point(by_image[weakest][cursor[weakest]++].point);
        }
    }
    for (const auto& candidate : ranked) {
        if (selected_count >= max_landmarks) break;
        add_point(candidate.point);
    }

    std::vector<int> remap(points.size(), -1);
    std::vector<Point3D> compact_points;
    compact_points.reserve(selected_count);
    for (size_t p = 0; p < points.size(); ++p) if (selected[p]) {
        remap[p] = static_cast<int>(compact_points.size());
        compact_points.push_back(points[p]);
    }
    std::vector<Observation> compact_observations;
    compact_observations.reserve(observations.size());
    for (auto obs : observations) if (obs.p >= 0 && obs.p < static_cast<int>(remap.size()) && remap[obs.p] >= 0) {
        obs.p = remap[obs.p]; compact_observations.push_back(obs);
    }
    points.swap(compact_points); observations.swap(compact_observations);
    int minimum_coverable = std::numeric_limits<int>::max();
    for (size_t f = 0; f < cameras.size(); ++f)
        if (coverable[f]) minimum_coverable = std::min(minimum_coverable, image_counts[f]);
    if (minimum_coverable == std::numeric_limits<int>::max()) minimum_coverable = 0;
    stats["selected_landmarks"] = points.size();
    stats["selected_observations"] = observations.size();
    stats["coverable_images"] = std::count(coverable.begin(), coverable.end(), true);
    stats["covered_images"] = std::count_if(image_counts.begin(), image_counts.end(), [](int n) { return n > 0; });
    stats["minimum_selected_landmarks_per_coverable_image"] = minimum_coverable;
    stats["mean_selected_landmarks_per_image"] = image_counts.empty() ? 0.0 : static_cast<double>(observations.size()) / image_counts.size();
    return stats;
}

std::pair<int, int> read_png_size(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    unsigned char header[24];
    if (!in.read(reinterpret_cast<char*>(header), sizeof(header)) ||
        std::string(reinterpret_cast<char*>(header), 8) != "\x89PNG\r\n\x1a\n" ||
        std::string(reinterpret_cast<char*>(header) + 12, 4) != "IHDR")
        throw std::runtime_error("Cannot read PNG dimensions from " + path.string());
    auto be32 = [&](int o) {
        return (header[o] << 24) | (header[o + 1] << 16) | (header[o + 2] << 8) | header[o + 3];
    };
    return {be32(16), be32(20)};
}

void save_outputs(
    const fs::path& output_dir,
    const fs::path& data_dir,
    std::vector<CameraState>& cameras,
    std::vector<Point3D>& points,
    const std::vector<Observation>& observations,
    const Mat3& K,
    const Json& metrics)
{
    fs::create_directories(output_dir / "sparse" / "0");

    std::ofstream poses_file(output_dir / "poses_optimized_tum.txt");
    std::ofstream images_file(output_dir / "sparse" / "0" / "images.txt");
    std::ofstream cameras_file(output_dir / "sparse" / "0" / "cameras.txt");
    std::ofstream points_file(output_dir / "sparse" / "0" / "points3D.txt");
    std::ofstream ply_file(output_dir / "landmarks_optimized.ply");

    const auto [width, height] = read_png_size(data_dir / "undistorted" / cameras.front().name);
    cameras_file << "1 PINHOLE " << width << ' ' << height << ' '
                 << K(0, 0) << ' ' << K(1, 1) << ' ' << K(0, 2) << ' ' << K(1, 2) << '\n';

    std::vector<std::vector<const Observation*>> image_observations(cameras.size());
    std::vector<std::vector<std::pair<int, int>>> point_tracks(points.size());
    for (const auto& observation : observations) {
        if (observation.f < 0 || observation.f >= static_cast<int>(cameras.size()) ||
            observation.p < 0 || observation.p >= static_cast<int>(points.size())) continue;
        image_observations[observation.f].push_back(&observation);
    }

    std::vector<int> image_ids(cameras.size(), -1);
    int next_image_id = 1;
    for (size_t i = 0; i < cameras.size(); ++i)
        if (!image_observations[i].empty()) image_ids[i] = next_image_id++;

    std::vector<int> point_ids(points.size(), -1);
    int next_point_id = 1;
    for (const auto& observation : observations)
        if (observation.p >= 0 && observation.p < static_cast<int>(points.size()) &&
            observation.f >= 0 && observation.f < static_cast<int>(cameras.size()) &&
            image_ids[observation.f] > 0 && point_ids[observation.p] < 0)
            point_ids[observation.p] = next_point_id++;

    for (size_t i = 0; i < cameras.size(); ++i) {
        auto& camera = cameras[i];
        poses_file << std::setprecision(16)
                   << camera.ts << ' '
                   << camera.t[0] << ' ' << camera.t[1] << ' ' << camera.t[2] << ' '
                   << camera.q[1] << ' ' << camera.q[2] << ' ' << camera.q[3] << ' ' << camera.q[0] << '\n';

        if (image_ids[i] < 0) continue;

        Quat q(camera.q[0], camera.q[1], camera.q[2], camera.q[3]);
        Quat q_conj = q.conjugate();
        Vec3 translation = -(q_conj * Vec3(camera.t));
        images_file << image_ids[i] << ' ' << q_conj.w() << ' ' << q_conj.x() << ' ' << q_conj.y() << ' ' << q_conj.z() << ' '
                    << translation.transpose() << " 1 " << camera.name << '\n';
        auto& frame_observations = image_observations[i];
        for (size_t j = 0; j < frame_observations.size(); ++j) {
            const auto& observation = *frame_observations[j];
            if (j) images_file << ' ';
            images_file << observation.u << ' ' << observation.v << ' ' << point_ids[observation.p];
            point_tracks[observation.p].emplace_back(image_ids[i], static_cast<int>(j));
        }
        images_file << '\n';
    }

    ply_file << "ply\n"
             << "format ascii 1.0\n"
             << "element vertex " << points.size() << "\n"
             << "property float x\n"
             << "property float y\n"
             << "property float z\n"
             << "property uchar red\n"
             << "property uchar green\n"
             << "property uchar blue\n"
             << "end_header\n";

    for (size_t i = 0; i < points.size(); ++i) {
        auto& point = points[i];
        if (point_ids[i] > 0) {
            points_file << point_ids[i] << ' ' << point.x[0] << ' ' << point.x[1] << ' ' << point.x[2]
                        << ' ' << point.r << ' ' << point.g << ' ' << point.b << " 0";
            for (const auto& track : point_tracks[i]) points_file << ' ' << track.first << ' ' << track.second;
            points_file << '\n';
        }
        ply_file << point.x[0] << ' ' << point.x[1] << ' ' << point.x[2] << ' ' << point.r << ' ' << point.g << ' ' << point.b << '\n';
    }

    std::ofstream evaluation_file(output_dir / "evaluation.json");
    evaluation_file << std::setw(2) << metrics << '\n';
}

int main(int argc, char** argv)
{
    try {
        fs::path data_dir = "data";
        fs::path output_dir = "output/lio_camera_pose";
        fs::path intrinsics_path;
        int max_images = -1;
        int local_start = 0;
        bool run_global = true;
        bool staged = false;
        double max_reprojection_error = 2.0;
        int cleanup_iterations = 50;
        int max_landmarks = 0;
        int min_landmarks_per_image = 20;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--data") {
                data_dir = argv[++i];
            } else if (arg == "--output") {
                output_dir = argv[++i];
            } else if (arg == "--intrinsics") {
                intrinsics_path = argv[++i];
            } else if (arg == "--max-images") {
                max_images = std::stoi(argv[++i]);
            } else if (arg == "--local-start") {
                local_start = std::stoi(argv[++i]);
            } else if (arg == "--no-global") {
                run_global = false;
            } else if (arg == "--staged") {
                staged = true;
            } else if (arg == "--max-reprojection-error") {
                max_reprojection_error = std::stod(argv[++i]);
            } else if (arg == "--cleanup-iterations") {
                cleanup_iterations = std::stoi(argv[++i]);
            } else if (arg == "--max-landmarks") {
                max_landmarks = std::stoi(argv[++i]);
            } else if (arg == "--min-landmarks-per-image") {
                min_landmarks_per_image = std::stoi(argv[++i]);
            } else if (arg == "--translation-prior-weight") {
                g_translation_prior_weight = std::stod(argv[++i]);
            } else if (arg == "--rotation-prior-weight") {
                g_rotation_prior_weight = std::stod(argv[++i]);
            } else if (arg == "--no-robust-pose-prior") {
                g_robust_pose_prior = false;
            }
        }
        if (!(max_reprojection_error > 0.0) || cleanup_iterations < 0 ||
            max_landmarks < 0 || min_landmarks_per_image < 0)
            throw std::runtime_error("cleanup threshold must be positive and iterations non-negative");
        if (!(g_translation_prior_weight > 0.0) || !(g_rotation_prior_weight > 0.0))
            throw std::runtime_error("pose prior weights must be positive");

        if (intrinsics_path.empty()) intrinsics_path = data_dir / "intrinsics.txt";
        Mat3 K = read_intrinsics(intrinsics_path);
        const Mat3 initial_K = K;
        auto cameras = read_camera_states(output_dir / "poses_lio_prior_tum.txt", data_dir / "timestamps.txt");
        if (max_images > 0 && max_images < static_cast<int>(cameras.size())) {
            cameras.resize(max_images);
        }
        const fs::path initial_poses = output_dir / "poses_initial_tum.txt";
        if (fs::exists(initial_poses)) initialize_camera_states(initial_poses, cameras);

        auto points = read_point_cloud(output_dir / "sparse_points.ply");
        auto observations = read_observations(output_dir / "tracks.csv");
        auto heldout_matches = read_heldout_matches(output_dir / "heldout.csv");
        const Json landmark_selection = select_landmarks_for_ba(
            points, observations, cameras, K, max_landmarks, min_landmarks_per_image);
        std::cout << "LANDMARK_SELECTION " << landmark_selection.dump() << '\n';

        int next_periodic = 100;
        for (int start = local_start; start < static_cast<int>(cameras.size()); start += 10) {
            const int end = std::min(static_cast<int>(cameras.size()), start + 15);
            run_local_bundle_adjustment(cameras, points, observations, K, start, end, 25, "LOCAL");
            if (staged && end >= next_periodic && end < static_cast<int>(cameras.size())) {
                run_local_bundle_adjustment(cameras, points, observations, K, 0, end, 40, "PERIODIC_GLOBAL");
                next_periodic += 100;
            }
        }

        size_t post_ba_pruned = 0;
        if (run_global) {
            run_local_bundle_adjustment(cameras, points, observations, K, 0, static_cast<int>(cameras.size()), 100, "GLOBAL");
            if (staged) {
                double k[4]={K(0,0),K(1,1),K(0,2),K(1,2)};
                const double prior[4]={initial_K(0,0),initial_K(1,1),initial_K(0,2),initial_K(1,2)};
                run_intrinsic_bundle_adjustment(cameras,points,observations,k,prior);
                K(0,0)=k[0]; K(1,1)=k[1]; K(0,2)=k[2]; K(1,2)=k[3];
                run_local_bundle_adjustment(cameras,points,observations,K,0,static_cast<int>(cameras.size()),100,"FINAL_GLOBAL");
                std::ofstream f(output_dir/"intrinsics_refined.txt");
                f<<std::setprecision(16)<<K(0,0)<<", 0, "<<K(0,2)<<"\n0, "<<K(1,1)<<", "<<K(1,2)<<"\n0, 0, 1\n";
            }
            auto cleaned = prune_observations(cameras, points, observations, K, max_reprojection_error);
            post_ba_pruned = observations.size() - cleaned.size();
            std::cout << "POST_BA_PRUNE removed=" << post_ba_pruned << " retained=" << cleaned.size() << '\n';
            observations.swap(cleaned);
            if (cleanup_iterations > 0 && !observations.empty()) {
                run_local_bundle_adjustment(cameras, points, observations, K, 0,
                    static_cast<int>(cameras.size()), cleanup_iterations, "CLEANUP_GLOBAL");
                cleaned = prune_observations(cameras, points, observations, K, max_reprojection_error);
                post_ba_pruned += observations.size() - cleaned.size();
                observations.swap(cleaned);
                std::cout << "FINAL_TARGET_PRUNE total_removed=" << post_ba_pruned
                          << " retained=" << observations.size() << '\n';
            }
            write_observations(output_dir / "tracks.csv", observations);
        }

        std::vector<double> reprojection_errors;
        std::vector<double> heldout_errors;
        std::vector<double> translation_deltas;
        std::vector<double> rotation_deltas;

        for (const auto& obs : observations) {
            reprojection_errors.push_back(reprojection_error(cameras[obs.f], points[obs.p], obs, K));
        }

        for (const auto& match : heldout_matches) {
            heldout_errors.push_back(epipolar_error(cameras[match.a], cameras[match.b], match, K));
        }

        for (const auto& camera : cameras) {
            const Vec3 translation_delta = Vec3(camera.t) - Vec3(camera.pt);
            translation_deltas.push_back(translation_delta.norm());

            const Quat q(camera.q[0], camera.q[1], camera.q[2], camera.q[3]);
            const Quat prior_q(camera.pq[0], camera.pq[1], camera.pq[2], camera.pq[3]);
            const double angle_deg = 2.0 * std::acos(std::clamp(std::abs(q.dot(prior_q)), 0.0, 1.0)) * 180.0 / M_PI;
            rotation_deltas.push_back(angle_deg);
        }

        std::vector<char> active_landmark_mask(points.size(), false);
        std::vector<char> active_image_mask(cameras.size(), false);
        for (const auto& obs : observations)
            if (obs.p >= 0 && obs.p < static_cast<int>(points.size()) &&
                obs.f >= 0 && obs.f < static_cast<int>(cameras.size())) {
                active_landmark_mask[obs.p] = true;
                active_image_mask[obs.f] = true;
            }
        const size_t active_landmarks = std::count(active_landmark_mask.begin(), active_landmark_mask.end(), true);
        const size_t sparse_export_images = std::count(active_image_mask.begin(), active_image_mask.end(), true);
        const double reprojection_median = quantile(reprojection_errors, 0.5);
        const double reprojection_p90 = quantile(reprojection_errors, 0.9);
        const double reprojection_p95 = quantile(reprojection_errors, 0.95);
        const double reprojection_max = quantile(reprojection_errors, 1.0);
        double time_offset_seconds = 0.0;
        std::ifstream initialization_metrics_file(output_dir / "metrics.json");
        if (initialization_metrics_file) {
            Json initialization_metrics;
            initialization_metrics_file >> initialization_metrics;
            if (initialization_metrics.contains("time_offset_seconds"))
                time_offset_seconds = initialization_metrics["time_offset_seconds"].get<double>();
        }

        Json metrics = {
            {"pass", false},
            {"images", cameras.size()},
            {"sparse_export_images", sparse_export_images},
            {"sparse_excluded_images", cameras.size() - sparse_export_images},
            {"landmarks", points.size()},
            {"active_landmarks", active_landmarks},
            {"observations", observations.size()},
            {"heldout_matches", heldout_matches.size()},
            {"reprojection_median_px", reprojection_median},
            {"post_ba_pruned_observations", post_ba_pruned},
            {"reprojection_p90_px", reprojection_p90},
            {"reprojection_p95_px", reprojection_p95},
            {"reprojection_max_px", reprojection_max},
            {"reprojection_target_px", max_reprojection_error},
            {"reprojection_target_met", reprojection_max < max_reprojection_error},
            {"heldout_epipolar_median_px", quantile(heldout_errors, 0.5)},
            {"heldout_epipolar_p90_px", quantile(heldout_errors, 0.9)},
            {"lio_translation_delta_median_m", quantile(translation_deltas, 0.5)},
            {"lio_rotation_delta_median_deg", quantile(rotation_deltas, 0.5)},
            {"time_offset_seconds", time_offset_seconds},
            {"translation_prior_weight", g_translation_prior_weight},
            {"rotation_prior_weight", g_rotation_prior_weight},
            {"robust_pose_prior", g_robust_pose_prior},
            {"intrinsics_fx",K(0,0)}, {"intrinsics_fy",K(1,1)},
            {"intrinsics_cx",K(0,2)}, {"intrinsics_cy",K(1,2)},
            {"intrinsics_refined",staged},
            {"landmark_selection",landmark_selection},
            {"accuracy_scope", "held-out consistency and LIO agreement; absolute accuracy requires independent ground truth"}
        };

        metrics["pass"] =
            points.size() >= 100 &&
            heldout_matches.size() >= 30 &&
            reprojection_max < max_reprojection_error &&
            quantile(reprojection_errors, 0.5) <= 2.0 &&
            quantile(heldout_errors, 0.5) <= 2.0 &&
            quantile(heldout_errors, 0.9) <= 5.0 &&
            quantile(translation_deltas, 0.5) <= 0.3 &&
            quantile(rotation_deltas, 0.5) <= 3.0;

        save_outputs(output_dir, data_dir, cameras, points, observations, K, metrics);
        std::cout << metrics.dump(2) << '\n';
        return metrics["pass"] ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << '\n';
        return 1;
    }
}
