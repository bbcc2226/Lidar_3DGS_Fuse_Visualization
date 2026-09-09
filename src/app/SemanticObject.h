#pragma once

#include <QString>
#include <QStringList>
#include <QVector3D>
#include <vector>

enum class SemanticReviewStatus { Unverified, Confirmed, Uncertain, Incorrect };

struct SemanticObservation
{
    QString image;
    QVector3D camera_world;
    double visibility = 0.0;
};

struct SemanticObject
{
    int id = -1;
    QString name;
    QString description;
    QString confidence;
    double confidence_score = 0.0;
    int observation_count = 0;
    QVector3D position_world;
    QVector3D bounds_min_world;
    QVector3D bounds_max_world;
    QStringList supporting_images;
    std::vector<SemanticObservation> observations;
    QVector3D representative_camera_world;
    double representative_visibility = 0.0;
    bool has_representative_camera = false;
    SemanticReviewStatus review = SemanticReviewStatus::Unverified;
};
