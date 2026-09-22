#pragma once

#include <QElapsedTimer>
#include <QMatrix4x4>
#include <QObject>
#include <QTimer>
#include <QVector3D>

#include <string>
#include <vector>

class RobotPathPlayer final : public QObject
{
    Q_OBJECT

public:
    explicit RobotPathPlayer(QObject* parent = nullptr);

    bool loadPath(const QString& path);
    bool setPlannedPath(const std::vector<QVector3D>& world_points,
                        const QVector3D& final_look_target_world);
    void setWorldToAlignedTransform(const QMatrix4x4& transform);
    void setSpeedMetersPerSecond(float speed);
    void togglePlayPause();
    void stop();
    void clearPath();

    bool hasPath() const { return !world_points_.empty(); }
    bool usesRecordedHeight() const { return uses_recorded_height_; }
    bool isPlaying() const { return timer_.isActive(); }
    const std::vector<QVector3D>& worldPoints() const { return world_points_; }
    QVector3D currentWorldPosition() const;
    const QString& lastError() const { return last_error_; }

signals:
    void poseChanged(const QVector3D& aligned_position,
                     float yaw_degrees, float pitch_degrees);
    void progressChanged(float progress);
    void playbackFinished();
    void playbackStateChanged(const QString& text);

private:
    QVector3D sampleWorldPosition(double distance) const;
    float desiredYaw(double distance) const;
    void updateFrame();
    void emitPose(bool initialize_yaw);
    void rebuildDistances();

    std::vector<QVector3D> world_points_;
    std::vector<double> cumulative_distance_;
    QMatrix4x4 world_to_aligned_;
    QTimer timer_;
    QElapsedTimer elapsed_;
    QString last_error_;
    double distance_ = 0.0;
    double total_distance_ = 0.0;
    float speed_ = 1.0f / 6.0f;
    float yaw_degrees_ = 0.0f;
    float pitch_degrees_ = 0.0f;
    QVector3D final_look_target_world_;
    bool has_final_look_target_ = false;
    bool final_turn_active_ = false;
    bool uses_recorded_height_ = false;
};
