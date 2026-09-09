#include "ViewerController.h"

#include "OpenGLWidget.h"
#include "OrientationGizmo.h"
#include "RobotPathPlayer.h"

#include <QApplication>
#include <QDir>
#include <QFileDialog>
#include <QKeyEvent>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMouseEvent>
#include <QSaveFile>
#include <QFile>
#include <QtMath>
#include <QWheelEvent>
#include <QWidget>
#include <QVariantAnimation>
#include <QEasingCurve>
#include <QDateTime>
#include <QFileInfo>
#include <QPainterPath>
#include <QPolygonF>
#include <QVector2D>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>

namespace
{
constexpr float kRotationDegreesPerPixel = 0.5f;
constexpr float kConstrainedRotationDegreesPerPixel = 0.15f;
constexpr float kSemanticRotationDegreesPerPixel = 0.08f;
constexpr float kInitialCameraDistance = 3.0f;
constexpr float kNavigateInitialCameraDistance = 0.65f;
constexpr float kMinimumCameraDistance = 0.02f;
constexpr float kMaximumCameraDistance = 100.0f;
constexpr float kDollyFactorPerWheelStep = 0.72f;
constexpr float kPanSensitivity = 1.75f;
constexpr float kHalfVerticalFieldOfViewRadians = 22.5f * 3.14159265f / 180.0f;
constexpr int kConstrainedDragThresholdPixels = 6;
constexpr float kVerticalMoveFraction = 0.05f;

bool isGoodSemanticViewpoint(const SemanticObject& object,
                             const QVector3D& current_aligned,
                             const QMatrix4x4& world_to_aligned,
                             float units_per_meter,
                             float preferred_distance_meters)
{
    const QVector3D target=world_to_aligned.map(object.position_world);
    QVector3D current_side=current_aligned-target;
    current_side.setZ(0.0f);
    const float distance_m=current_side.length()/units_per_meter;
    if(distance_m<0.75f*preferred_distance_meters ||
       distance_m>1.35f*preferred_distance_meters ||
       current_side.lengthSquared()<1.0e-8f) return false;
    for(const SemanticObservation& observation:object.observations){
        if(observation.visibility<0.25) continue;
        QVector3D observation_offset=
            current_aligned-world_to_aligned.map(observation.camera_world);
        observation_offset.setZ(0.0f);
        // Suppress movement only when the robot is actually near a known-good
        // viewpoint. Merely being on the same side can span an entire room and
        // previously made the mug endpoint look valid for the TV.
        if(observation_offset.length()/units_per_meter<=0.50f)
            return true;
    }
    return false;
}
}

ViewerController::ViewerController(OpenGLWidget* viewer, QObject* parent)
    : QObject(parent), viewer_(viewer)
{
    robot_path_player_ = new RobotPathPlayer(this);
    arrival_view_animation_ = new QVariantAnimation(this);
    // Arrival is deliberately slower than path steering: a person naturally
    // settles their gaze after stopping instead of snapping toward the target.
    arrival_view_animation_->setDuration(1100);
    arrival_view_animation_->setStartValue(0.0);
    arrival_view_animation_->setEndValue(1.0);
    arrival_view_animation_->setEasingCurve(QEasingCurve::InOutCubic);
    connect(arrival_view_animation_, &QVariantAnimation::valueChanged, this,
            [this](const QVariant& value) {
                if (!has_navigation_target_) return;
                const float amount = value.toFloat();
                z_up_camera_position_ = arrival_view_start_position_ * (1.0f - amount) +
                    arrival_view_end_position_ * amount;
                yaw_degrees_ = arrival_view_start_yaw_ * (1.0f - amount) +
                    arrival_view_end_yaw_ * amount;
                pitch_degrees_ = arrival_view_start_pitch_ * (1.0f - amount) +
                    arrival_view_end_pitch_ * amount;
                applyTransform();
            });
    connect(arrival_view_animation_, &QVariantAnimation::finished, this,
            [this]() { viewer_->setNavigationTargetTagVisible(true); });
    connect(robot_path_player_, &RobotPathPlayer::poseChanged, this,
            [this](const QVector3D& position, float yaw, float pitch) {
                robot_pose_initialized_ = true;
                constrained_z_up_navigation_ = true;
                path_editing_enabled_ = false;
                z_up_camera_position_ = position;
                z_up_camera_position_.setZ(
                    viewer_->robotEyeHeightAligned(
                        1.0f, z_up_camera_position_.z()));
                yaw_degrees_ = yaw;
                pitch_degrees_ = pitch;
                viewer_->setPathEditMode(false);
                viewer_->setZUpGizmo(true);
                applyTransform();
            });
    connect(robot_path_player_, &RobotPathPlayer::playbackStateChanged, this,
            [this](const QString& text) {
                emit robotPlaybackStatusChanged(text, QString());
            });
    connect(robot_path_player_, &RobotPathPlayer::playbackFinished, this,
            [this]() {
                beginArrivalView();
            });
    viewer_->installEventFilter(this);
    applyTransform();
}

void ViewerController::restoreNavigationView()
{
    arrival_view_animation_->stop();
    viewer_->setVerticalFieldOfView(45.0f);
    pitch_degrees_ = 0.0f;
    if (constrained_z_up_navigation_)
        z_up_camera_position_.setZ(viewer_->robotEyeHeightAligned(
            1.0f, z_up_camera_position_.z()));
    applyTransform();
}

void ViewerController::beginArrivalView()
{
    if (!has_navigation_target_ || path_editing_enabled_ ||
        free_zone_editing_enabled_) return;
    arrival_view_start_position_ = z_up_camera_position_;
    arrival_view_start_yaw_ = yaw_degrees_;
    arrival_view_start_pitch_ = pitch_degrees_;
    // Keep the camera at its actual eye position. Only yaw and pitch ease
    // toward the object; changing height here caused the old ceiling sweep.
    arrival_view_end_position_ = z_up_camera_position_;
    const QVector3D direction =
        (navigation_target_center_aligned_ - arrival_view_end_position_).normalized();
    const float target_yaw =
        qRadiansToDegrees(std::atan2(direction.x(), direction.y()));
    arrival_view_end_pitch_ = -qRadiansToDegrees(std::asin(
        std::clamp(direction.z(), -1.0f, 1.0f)));
    // Interpolate the shortest turn across the -180/180 degree boundary.
    float yaw_delta = std::fmod(target_yaw - arrival_view_start_yaw_ + 540.0f,
                                360.0f) - 180.0f;
    arrival_view_end_yaw_ = arrival_view_start_yaw_ + yaw_delta;
    arrival_view_animation_->start();
}

bool ViewerController::loadSemanticDatabaseFile(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        emit semanticDatabaseChanged("Cannot open semantic database: " + file.errorString(), path);
        return false;
    }
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !document.isObject()) {
        emit semanticDatabaseChanged("Invalid semantic database: " + error.errorString(), path);
        return false;
    }
    std::vector<SemanticObject> loaded;
    const QJsonArray objects = document.object().value("objects").toArray();
    semantic_aliases_.clear();
    const QJsonObject aliases = document.object().value("aliases").toObject();
    for (auto it = aliases.begin(); it != aliases.end(); ++it)
        semantic_aliases_.insert(it.key().trimmed().toLower(), it.value().toString());
    loaded.reserve(objects.size());
    const auto vector3 = [](const QJsonValue& value, QVector3D& result) {
        const QJsonArray a = value.toArray();
        if (a.size() != 3) return false;
        result = QVector3D(a[0].toDouble(), a[1].toDouble(), a[2].toDouble());
        return true;
    };
    for (const QJsonValue& value : objects) {
        const QJsonObject json = value.toObject();
        SemanticObject object;
        object.id = json.value("object_id").toInt(-1);
        object.name = json.value("canonical_class").toString();
        object.description = json.value("description").toString();
        object.confidence = json.value("confidence").toString();
        object.confidence_score = json.value("confidence_score").toDouble();
        object.observation_count = json.value("observation_count").toInt();
        if (object.id < 0 || object.name.isEmpty() ||
            !vector3(json.value("bounds_min_world_m"), object.bounds_min_world) ||
            !vector3(json.value("bounds_max_world_m"), object.bounds_max_world)) continue;
        if (!vector3(json.value("position_world_m"), object.position_world))
            object.position_world = 0.5f *
                (object.bounds_min_world + object.bounds_max_world);
        double best_visibility = -1.0;
        for (const QJsonValue& view : json.value("views").toArray()) {
            const QJsonObject view_json = view.toObject();
            const QString image = view_json.value("image").toString();
            if (!image.isEmpty()) object.supporting_images.append(image);
            const double visibility = view_json.value("visibility_score").toDouble();
            QVector3D camera_position;
            const QJsonObject camera_pose = view_json.value("camera_pose").toObject();
            const bool has_camera = vector3(
                camera_pose.value("position_world_m"), camera_position);
            if (has_camera) {
                object.observations.push_back({image, camera_position, visibility});
            }
            if (visibility > best_visibility && has_camera) {
                best_visibility = visibility;
                object.representative_camera_world = camera_position;
                object.representative_visibility = visibility;
                object.has_representative_camera = true;
            }
        }
        loaded.push_back(std::move(object));
    }
    semantic_objects_ = std::move(loaded);
    semantic_database_path_ = path;
    semantic_review_path_ = QFileInfo(path).absolutePath() + "/" +
        QFileInfo(path).completeBaseName() + ".reviews.json";
    semantic_reviews_dirty_ = false;
    loadSemanticReviewsFile(semantic_review_path_);
    rebuildSpatialRelations();
    selected_semantic_object_id_ = -1;
    viewer_->setSemanticObjects(semantic_objects_);
    int reviewed = 0;
    for (const SemanticObject& object : semantic_objects_)
        if (object.review != SemanticReviewStatus::Unverified) ++reviewed;
    emit semanticDatabaseChanged(
        QString("Loaded %1 semantic objects (%2 reviewed)")
            .arg(semantic_objects_.size()).arg(reviewed), path);
    emit semanticSelectionChanged(-1);
    return true;
}

void ViewerController::rebuildSpatialRelations()
{
    spatial_relations_.clear();
    const QMatrix4x4 transform = viewer_->sceneWorldToAlignedTransform();
    const float units_per_meter = std::max(
        transform.mapVector(QVector3D(1, 0, 0)).length(), 1.0e-5f);
    struct Box { int id; QString name; QVector3D min, max, center; };
    std::vector<Box> boxes;
    for (const SemanticObject& object : semantic_objects_) {
        if (object.review != SemanticReviewStatus::Confirmed) continue;
        QVector3D minimum(
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max());
        QVector3D maximum = -minimum;
        for (int corner = 0; corner < 8; ++corner) {
            const QVector3D world(
                corner & 1 ? object.bounds_max_world.x()
                           : object.bounds_min_world.x(),
                corner & 2 ? object.bounds_max_world.y()
                           : object.bounds_min_world.y(),
                corner & 4 ? object.bounds_max_world.z()
                           : object.bounds_min_world.z());
            const QVector3D aligned = transform.map(world);
            minimum.setX(std::min(minimum.x(), aligned.x()));
            minimum.setY(std::min(minimum.y(), aligned.y()));
            minimum.setZ(std::min(minimum.z(), aligned.z()));
            maximum.setX(std::max(maximum.x(), aligned.x()));
            maximum.setY(std::max(maximum.y(), aligned.y()));
            maximum.setZ(std::max(maximum.z(), aligned.z()));
        }
        boxes.push_back({object.id, object.name, minimum, maximum,
                         transform.map(object.position_world)});
    }
    for (const Box& s : boxes) for (const Box& r : boxes) {
        if (s.id==r.id) continue;
        const float dx=std::max({0.0f,r.min.x()-s.max.x(),s.min.x()-r.max.x()});
        const float dy=std::max({0.0f,r.min.y()-s.max.y(),s.min.y()-r.max.y()});
        const float gap=std::hypot(dx,dy);
        if(gap<0.75f*units_per_meter) spatial_relations_.push_back({s.id,r.id,"near",1.0f-gap/(.75f*units_per_meter)});
        if(gap<0.25f*units_per_meter) spatial_relations_.push_back({s.id,r.id,"next to",1.0f-gap/(.25f*units_per_meter)});
        // A transformed axis-aligned box is slightly larger than the actual
        // shelf footprint. Require a small interior margin so nearby objects
        // touching that expanded boundary (for example the water-filter mug)
        // are not falsely classified as being on the shelf.
        const float relation_inset = 0.04f * units_per_meter;
        const float inset_x = std::min(
            relation_inset, 0.2f * (r.max.x() - r.min.x()));
        const float inset_y = std::min(
            relation_inset, 0.2f * (r.max.y() - r.min.y()));
        const bool horizontal_containment =
            s.center.x() >= r.min.x() + inset_x &&
            s.center.x() <= r.max.x() - inset_x &&
            s.center.y() >= r.min.y() + inset_y &&
            s.center.y() <= r.max.y() - inset_y;
        const bool vertical_overlap = s.max.z() >= r.min.z() &&
            s.min.z() <= r.max.z();
        // Shelf/counter boxes are often coarse and enclose their contents, so
        // do not require the contents to touch the box's top face exactly.
        if (horizontal_containment && vertical_overlap)
            spatial_relations_.push_back({s.id,r.id,"on",0.75f});
        if(s.max.x()<r.min.x()) spatial_relations_.push_back({s.id,r.id,"left of",0.8f});
        if(s.min.x()>r.max.x()) spatial_relations_.push_back({s.id,r.id,"right of",0.8f});
    }
}

void ViewerController::loadSemanticReviewsFile(const QString& path)
{
    QFile file(path);
    if (!file.exists() || !file.open(QIODevice::ReadOnly)) return;
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !document.isObject()) return;
    const QJsonArray reviews = document.object().value("reviews").toArray();
    for (const QJsonValue& value : reviews) {
        const QJsonObject json = value.toObject();
        const int id = json.value("object_id").toInt(-1);
        const QString status = json.value("status").toString();
        SemanticReviewStatus parsed = SemanticReviewStatus::Unverified;
        if (status == "confirmed") parsed = SemanticReviewStatus::Confirmed;
        else if (status == "uncertain") parsed = SemanticReviewStatus::Uncertain;
        else if (status == "incorrect") parsed = SemanticReviewStatus::Incorrect;
        for (SemanticObject& object : semantic_objects_)
            if (object.id == id) { object.review = parsed; break; }
    }
}

void ViewerController::saveSemanticReviews(QWidget* dialog_parent, bool save_as)
{
    if (semantic_objects_.empty()) {
        emit semanticDatabaseChanged("No semantic objects to save", QString());
        return;
    }
    QString path = semantic_review_path_;
    if (save_as || path.isEmpty()) {
        QFileDialog dialog(dialog_parent, "Save semantic verification results",
                           path.isEmpty() ? QStringLiteral(PROJECT_ROOT_DIR) +
                               "/data/semantic_reviews.json" : path);
        dialog.setAcceptMode(QFileDialog::AcceptSave);
        dialog.setFileMode(QFileDialog::AnyFile);
        dialog.setNameFilter("JSON review files (*.json)");
        dialog.setDefaultSuffix("json");
        dialog.setOption(QFileDialog::DontUseNativeDialog, true);
        if (dialog.exec() != QDialog::Accepted || dialog.selectedFiles().isEmpty()) return;
        path = dialog.selectedFiles().constFirst();
    }

    const auto statusName = [](SemanticReviewStatus status) {
        switch (status) {
        case SemanticReviewStatus::Confirmed: return QStringLiteral("confirmed");
        case SemanticReviewStatus::Uncertain: return QStringLiteral("uncertain");
        case SemanticReviewStatus::Incorrect: return QStringLiteral("incorrect");
        default: return QStringLiteral("unverified");
        }
    };
    QJsonArray reviews;
    int reviewed_count = 0;
    for (const SemanticObject& object : semantic_objects_) {
        QJsonObject review;
        review["object_id"] = object.id;
        review["canonical_class"] = object.name;
        review["status"] = statusName(object.review);
        reviews.append(review);
        if (object.review != SemanticReviewStatus::Unverified) ++reviewed_count;
    }
    QJsonObject root;
    root["version"] = 1;
    root["source_database"] = semantic_database_path_;
    root["saved_at_utc"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    root["object_count"] = static_cast<int>(semantic_objects_.size());
    root["reviewed_count"] = reviewed_count;
    root["reviews"] = reviews;
    QSaveFile output(path);
    if (!output.open(QIODevice::WriteOnly)) {
        emit semanticDatabaseChanged("Cannot save reviews: " + output.errorString(), QString());
        return;
    }
    const QByteArray data = QJsonDocument(root).toJson(QJsonDocument::Indented);
    if (output.write(data) != data.size() || !output.commit()) {
        emit semanticDatabaseChanged("Failed to save reviews: " + output.errorString(), QString());
        return;
    }
    semantic_review_path_ = path;
    semantic_reviews_dirty_ = false;
    emit semanticDatabaseChanged(
        QString("Saved %1 reviewed objects to %2").arg(reviewed_count).arg(path), QString());
}

void ViewerController::loadSemanticDatabase(QWidget* dialog_parent)
{
    const QString path = QFileDialog::getOpenFileName(
        dialog_parent, "Open semantic object database",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data/scementic_objects",
        "JSON databases (*.json);;All files (*)");
    if (!path.isEmpty()) loadSemanticDatabaseFile(path);
}

void ViewerController::loadDefaultSemanticDatabase()
{
    loadSemanticDatabaseFile(QStringLiteral(PROJECT_ROOT_DIR) +
        "/data/scementic_objects/navigation_semantic_db.json");
}

void ViewerController::setSemanticObjectsVisible(bool visible)
{
    viewer_->setSemanticObjectsVisible(visible);
}

void ViewerController::setSemanticClassFilter(const QString& filter)
{
    viewer_->setSemanticClassFilter(filter);
}

void ViewerController::selectSemanticObject(int object_id)
{
    selected_semantic_object_id_ = object_id;
    viewer_->setSelectedSemanticObject(object_id);
    emit semanticSelectionChanged(object_id);
    if (!semantic_verification_mode_) return;
    for (const SemanticObject& object : semantic_objects_) {
        if (object.id != object_id || !object.has_representative_camera) continue;
        const QMatrix4x4 world_to_aligned = viewer_->sceneWorldToAlignedTransform();
        const QVector3D camera = world_to_aligned.map(object.representative_camera_world);
        const QVector3D center_world = 0.5f *
            (object.bounds_min_world + object.bounds_max_world);
        const QVector3D center = world_to_aligned.map(center_world);
        QVector3D forward = (center - camera).normalized();
        if (forward.lengthSquared() < 1.0e-8f) break;
        constrained_z_up_navigation_ = true;
        path_editing_enabled_ = false;
        z_up_camera_position_ = camera;
        yaw_degrees_ = qRadiansToDegrees(std::atan2(forward.x(), forward.y()));
        pitch_degrees_ = -qRadiansToDegrees(std::asin(
            std::clamp(forward.z(), -1.0f, 1.0f)));
        viewer_->setPathEditMode(false);
        viewer_->setZUpGizmo(true);
        applyTransform();
        break;
    }
}

void ViewerController::setSemanticVerificationMode(bool enabled)
{
    semantic_verification_mode_ = enabled;
    viewer_->setSemanticSelectedOnly(true);
}

void ViewerController::rotateSemanticView(float delta_degrees)
{
    if (!semantic_verification_mode_) return;
    constrained_z_up_navigation_ = true;
    yaw_degrees_ = std::fmod(yaw_degrees_ + delta_degrees, 360.0f);
    applyTransform();
}

bool ViewerController::planThroughFreeZone(
    const QString& canonical, const QString& requested_destination)
{
    if (!free_zone_closed_ || free_zone_vertices_.size() < 3) return false;
    QPainterPath free_area;
    free_area.setFillRule(Qt::WindingFill);
    std::vector<std::vector<QVector3D>> polygons = completed_free_zone_polygons_;
    polygons.push_back(free_zone_vertices_);
    for (const auto& vertices : polygons) {
        QPolygonF polygon;
        for (const QVector3D& p : vertices) polygon << QPointF(p.x(), p.y());
        QPainterPath component; component.addPolygon(polygon); component.closeSubpath();
        free_area = free_area.isEmpty() ? component : free_area.united(component);
    }
    const QRectF bounds = free_area.boundingRect();
    if (bounds.isEmpty()) return false;

    const QMatrix4x4 world_to_aligned = viewer_->sceneWorldToAlignedTransform();
    bool invertible = false;
    const QMatrix4x4 aligned_to_world = world_to_aligned.inverted(&invertible);
    if (!invertible) {
        emit navigationPlanStatusChanged("Free-zone transform is not invertible.");
        return true;
    }
    const float units_per_meter = std::max(
        world_to_aligned.mapVector(QVector3D(1, 0, 0)).length(), 1.0e-5f);
    const float clearance = 0.30f * units_per_meter;
    const float resolution = std::max(
        0.08f * units_per_meter,
        static_cast<float>(std::max(bounds.width(), bounds.height())) / 240.0f);
    const int columns = std::max(2, static_cast<int>(std::ceil(bounds.width() / resolution)) + 1);
    const int rows = std::max(2, static_cast<int>(std::ceil(bounds.height() / resolution)) + 1);
    const auto cell_point = [&](int cell) {
        return QPointF(bounds.left() + (cell % columns + 0.5) * resolution,
                       bounds.top() + (cell / columns + 0.5) * resolution);
    };
    std::vector<std::uint8_t> walkable(static_cast<std::size_t>(columns * rows), 0);
    static constexpr float directions[8][2] = {
        {1,0},{-1,0},{0,1},{0,-1},{0.7071f,0.7071f},{0.7071f,-0.7071f},
        {-0.7071f,0.7071f},{-0.7071f,-0.7071f}};
    for (int cell = 0; cell < columns * rows; ++cell) {
        const QPointF p = cell_point(cell);
        bool valid = free_area.contains(p);
        for (const auto& d : directions)
            valid = valid && free_area.contains(p + QPointF(d[0] * clearance,
                                                             d[1] * clearance));
        walkable[static_cast<std::size_t>(cell)] = valid ? 1 : 0;
    }
    const auto nearest_cell = [&](const QVector3D& point) {
        int best = -1; float best_squared = std::numeric_limits<float>::max();
        for (int cell = 0; cell < columns * rows; ++cell) {
            if (!walkable[static_cast<std::size_t>(cell)]) continue;
            const QPointF p = cell_point(cell);
            const float dx = static_cast<float>(p.x()) - point.x();
            const float dy = static_cast<float>(p.y()) - point.y();
            const float squared = dx * dx + dy * dy;
            if (squared < best_squared) { best_squared = squared; best = cell; }
        }
        return best;
    };

    QVector3D current_aligned;
    // A generated navigation route replaces RobotPathPlayer's points, so its
    // current position must not become the start of the next request.  Keep
    // navigation reproducible by using the start saved with the free zone
    // (originally the first point of the user's selected robot path).
    if (has_free_zone_default_start_)
        current_aligned = world_to_aligned.map(free_zone_default_start_world_);
    else if (!navigation_path_world_.empty())
        current_aligned = world_to_aligned.map(navigation_path_world_.front());
    else if (robot_path_player_->hasPath())
        current_aligned = world_to_aligned.map(robot_path_player_->worldPoints().front());
    else {
        QVector3D centroid;
        for (const QVector3D& point : free_zone_vertices_) centroid += point;
        current_aligned = centroid / static_cast<float>(free_zone_vertices_.size());
    }
    const int start_cell = nearest_cell(current_aligned);
    if (start_cell < 0) {
        emit navigationPlanStatusChanged(
            "The free zone has no cells remaining after robot-clearance erosion.");
        return true;
    }
    static constexpr int neighbors[8][2] = {
        {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    std::vector<std::uint8_t> reachable(walkable.size(), 0);
    std::queue<int> flood;
    reachable[static_cast<std::size_t>(start_cell)] = 1;
    flood.push(start_cell);
    while (!flood.empty()) {
        const int cell = flood.front(); flood.pop();
        const int x = cell % columns, y = cell / columns;
        for (const auto& n : neighbors) {
            const int nx = x + n[0], ny = y + n[1];
            if (nx < 0 || nx >= columns || ny < 0 || ny >= rows) continue;
            const int next = ny * columns + nx;
            if (!walkable[static_cast<std::size_t>(next)] ||
                reachable[static_cast<std::size_t>(next)]) continue;
            reachable[static_cast<std::size_t>(next)] = 1;
            flood.push(next);
        }
    }
    bool class_found = false;
    const SemanticObject* selected_object = nullptr;
    const SemanticObservation* selected_observation = nullptr;
    int goal_cell = -1;
    float best_score = std::numeric_limits<float>::max();
    float closest_reachable_error = std::numeric_limits<float>::max();
    float selected_view_offset_m = 0.0f;
    QVector3D object_center_aligned;
    for (const SemanticObject& object : semantic_objects_) {
        if (object.name.compare(canonical, Qt::CaseInsensitive) != 0) continue;
        if (preferred_navigation_object_id_ >= 0 &&
            object.id != preferred_navigation_object_id_) continue;
        if (!relation_target_ids_.empty() && !relation_target_ids_.count(object.id)) continue;
        class_found = true;
        if (object.review != SemanticReviewStatus::Confirmed) continue;
        const QVector3D center_world = object.position_world;
        const QVector3D center = world_to_aligned.map(center_world);
        const QVector3D size = object.bounds_max_world-object.bounds_min_world;
        const float preferred_meters = std::clamp(
            0.65f+0.5f*std::max(size.x(),size.y()),0.75f,1.25f);
        int candidate_cell = -1;
        float placement_error = std::numeric_limits<float>::max();
        float approach_distance = std::numeric_limits<float>::max();
        const QPointF start_point=cell_point(start_cell);
        const bool already_well_placed=isGoodSemanticViewpoint(
            object,QVector3D(start_point.x(),start_point.y(),center.z()),
            world_to_aligned,units_per_meter,preferred_meters);
        for (int cell = 0; cell < columns * rows; ++cell) {
            if (!reachable[static_cast<std::size_t>(cell)]) continue;
            const QPointF candidate = cell_point(cell);
            const float distance = static_cast<float>(std::hypot(
                candidate.x() - center.x(), candidate.y() - center.y())) /
                units_per_meter;
            const float travel_meters=static_cast<float>(std::hypot(
                candidate.x()-start_point.x(),candidate.y()-start_point.y()))/
                units_per_meter;
            const float stand_error=already_well_placed
                ? (cell==start_cell ? 0.0f : std::numeric_limits<float>::max())
                : std::abs(distance-preferred_meters)+0.20f*travel_meters;
            if (stand_error < placement_error) {
                placement_error=stand_error; approach_distance=distance;
                candidate_cell=cell;
            }
        }
        if (candidate_cell < 0) continue;
        closest_reachable_error = std::min(
            closest_reachable_error, approach_distance);
        const float quality_penalty=
            0.8f*static_cast<float>(1.0-object.confidence_score)+
            0.4f/std::sqrt(static_cast<float>(std::max(1,object.observation_count)));
        const float score=placement_error+quality_penalty;
        if (score < best_score) {
            best_score = score; selected_object = &object;
            selected_observation = nullptr; goal_cell = candidate_cell;
            object_center_aligned = center;
            selected_view_offset_m = approach_distance;
        }
    }
    if (!class_found) {
        emit navigationPlanStatusChanged("Object not found: " + requested_destination);
        return true;
    }
    if (!selected_object || goal_cell < 0) {
        emit navigationPlanStatusChanged(
            std::isfinite(closest_reachable_error)
                ? QString("Object exists, but its nearest start-connected safe cell is %1 m away. Check polygon connections or clearance.")
                    .arg(closest_reachable_error, 0, 'f', 2)
                : QString("No confirmed instance has a reachable free-zone viewing position."));
        return true;
    }
    struct QueueNode { float cost; int cell; };
    struct Greater { bool operator()(const QueueNode& a, const QueueNode& b) const {
        return a.cost > b.cost; }};
    const int cell_count = columns * rows;
    std::vector<float> cost(static_cast<std::size_t>(cell_count),
                            std::numeric_limits<float>::max());
    std::vector<int> parent(static_cast<std::size_t>(cell_count), -1);
    std::priority_queue<QueueNode, std::vector<QueueNode>, Greater> queue;
    cost[static_cast<std::size_t>(start_cell)] = 0.0f;
    queue.push({0.0f, start_cell});
    while (!queue.empty()) {
        const QueueNode node = queue.top(); queue.pop();
        if (node.cell == goal_cell) break;
        if (node.cost != cost[static_cast<std::size_t>(node.cell)]) continue;
        const int x = node.cell % columns, y = node.cell / columns;
        for (const auto& n : neighbors) {
            const int nx = x + n[0], ny = y + n[1];
            if (nx < 0 || nx >= columns || ny < 0 || ny >= rows) continue;
            const int next = ny * columns + nx;
            if (!walkable[static_cast<std::size_t>(next)]) continue;
            const float step = (n[0] && n[1]) ? 1.41421356f : 1.0f;
            const float next_cost = node.cost + step;
            if (next_cost < cost[static_cast<std::size_t>(next)]) {
                cost[static_cast<std::size_t>(next)] = next_cost;
                parent[static_cast<std::size_t>(next)] = node.cell;
                queue.push({next_cost, next});
            }
        }
    }
    if (parent[static_cast<std::size_t>(goal_cell)] < 0 && start_cell != goal_cell) {
        emit navigationPlanStatusChanged("No connected route exists inside the free zone.");
        return true;
    }
    std::vector<int> cells;
    for (int cell = goal_cell; cell >= 0; cell = parent[static_cast<std::size_t>(cell)]) {
        cells.push_back(cell); if (cell == start_cell) break;
    }
    std::reverse(cells.begin(), cells.end());
    const auto line_clear = [&](int from, int to) {
        const QPointF a = cell_point(from), b = cell_point(to);
        const int samples = std::max(1, static_cast<int>(std::ceil(
            std::hypot(b.x()-a.x(), b.y()-a.y()) / (resolution * 0.5))));
        for (int i = 0; i <= samples; ++i) {
            const float t = static_cast<float>(i) / samples;
            const double px = a.x()*(1-t)+b.x()*t;
            const double py = a.y()*(1-t)+b.y()*t;
            const int x = static_cast<int>(std::lround(
                (px - bounds.left()) / resolution - 0.5));
            const int y = static_cast<int>(std::lround(
                (py - bounds.top()) / resolution - 0.5));
            if (x < 0 || x >= columns || y < 0 || y >= rows ||
                !walkable[static_cast<std::size_t>(y * columns + x)]) return false;
        }
        return true;
    };
    std::vector<int> smooth_cells;
    for (std::size_t anchor = 0; anchor < cells.size();) {
        smooth_cells.push_back(cells[anchor]);
        if (anchor + 1 >= cells.size()) break;
        std::size_t farthest = anchor + 1;
        for (std::size_t candidate = anchor + 2; candidate < cells.size(); ++candidate)
            if (line_clear(cells[anchor], cells[candidate])) farthest = candidate;
        anchor = farthest;
    }
    std::vector<QVector3D> route_aligned;
    route_aligned.reserve(smooth_cells.size());
    const float route_z = current_aligned.z();
    for (int cell : smooth_cells) {
        const QPointF p = cell_point(cell);
        route_aligned.emplace_back(p.x(), p.y(), route_z);
    }
    std::vector<QVector3D> route_world;
    route_world.reserve(route_aligned.size());
    for (const QVector3D& p : route_aligned) route_world.push_back(aligned_to_world.map(p));
    const QVector3D navigation_target_aligned =
        world_to_aligned.map(selected_object->position_world);
    viewer_->setNavigationPlan(route_aligned, navigation_target_aligned,
        QString("%1 #%2").arg(selected_object->name).arg(selected_object->id));
    robot_path_player_->setWorldToAlignedTransform(world_to_aligned);
    const QVector3D object_center_world = selected_object->position_world;
    navigation_target_center_aligned_ = navigation_target_aligned;
    has_navigation_target_ = true;
    if (!robot_path_player_->setPlannedPath(route_world, object_center_world)) {
        emit navigationPlanStatusChanged("Could not start free-zone route.");
        return true;
    }
    selectSemanticObject(selected_object->id);
    setPathEditingEnabled(false);
    constrained_z_up_navigation_ = true;
    viewer_->setZUpGizmo(true);
    robot_path_player_->togglePlayPause();
    emit navigationPlanStatusChanged(
        QString("Free-zone navigation: %1 #%2 via %3 waypoints; view offset %4 m; source %5")
            .arg(selected_object->name).arg(selected_object->id)
            .arg(route_world.size()).arg(selected_view_offset_m, 0, 'f', 2)
            .arg(selected_observation->image));
    return true;
}

bool ViewerController::planThroughWalkableCells(
    const QString& canonical, const QString& requested_destination)
{
    if (walkable_cells_.empty()) return false;
    const QMatrix4x4 world_to_aligned = viewer_->sceneWorldToAlignedTransform();
    bool ok = false;
    const QMatrix4x4 aligned_to_world = world_to_aligned.inverted(&ok);
    if (!ok) return false;
    const float units_per_meter = std::max(
        world_to_aligned.mapVector(QVector3D(1,0,0)).length(), 1.0e-5f);
    const auto center = [this](const std::pair<int,int>& c) {
        const float x=(c.first+.5f)*walkable_cell_size_;
        const float y=(c.second+.5f)*walkable_cell_size_;
        const float cs=std::cos(walkable_grid_angle_radians_);
        const float sn=std::sin(walkable_grid_angle_radians_);
        return QVector3D(cs*x-sn*y, sn*x+cs*y, walkable_floor_z_);
    };
    const auto nearest = [&](const QVector3D& p,
                             const std::set<std::pair<int,int>>& allowed) {
        std::pair<int,int> result{}; float best = std::numeric_limits<float>::max();
        bool found = false;
        for (const auto& c : allowed) {
            const QVector3D q = center(c);
            const float d = QVector2D(q.x()-p.x(), q.y()-p.y()).lengthSquared();
            if (d < best) { best=d; result=c; found=true; }
        }
        return std::make_pair(found, result);
    };
    QVector3D start_position;
    if (!navigation_has_started_ && has_free_zone_default_start_)
        start_position = world_to_aligned.map(free_zone_default_start_world_);
    else if (!navigation_has_started_ && !navigation_path_world_.empty())
        start_position = world_to_aligned.map(navigation_path_world_.front());
    else
        start_position = world_to_aligned.map(robot_path_player_->currentWorldPosition());
    const auto start_result = nearest(start_position, walkable_cells_);
    if (!start_result.first) return false;
    const auto start = start_result.second;
    static constexpr int dirs[8][2] = {
        {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    std::set<std::pair<int,int>> reachable{start};
    std::queue<std::pair<int,int>> flood; flood.push(start);
    while (!flood.empty()) {
        const auto c=flood.front(); flood.pop();
        for (const auto& d:dirs) {
            const std::pair<int,int> n{c.first+d[0],c.second+d[1]};
            if (walkable_cells_.count(n) && reachable.insert(n).second) flood.push(n);
        }
    }
    const SemanticObject* selected = nullptr;
    std::pair<int,int> goal{}; QVector3D target; float best_score=std::numeric_limits<float>::max();
    bool class_found=false;
    for (const SemanticObject& object:semantic_objects_) {
        if (object.name.compare(canonical,Qt::CaseInsensitive)!=0) continue;
        if (preferred_navigation_object_id_>=0 &&
            object.id!=preferred_navigation_object_id_) continue;
        if (!relation_target_ids_.empty() && !relation_target_ids_.count(object.id)) continue;
        class_found=true; if (object.review!=SemanticReviewStatus::Confirmed) continue;
        const QVector3D object_center =
            world_to_aligned.map(object.position_world);
        const QVector3D size = object.bounds_max_world-object.bounds_min_world;
        const float preferred_meters = std::clamp(
            0.65f+0.5f*std::max(size.x(),size.y()),0.75f,1.25f);
        std::pair<int,int> candidate{};
        float best_stand_error=std::numeric_limits<float>::max();
        const bool already_well_placed=isGoodSemanticViewpoint(
            object,center(start),world_to_aligned,units_per_meter,
            preferred_meters);
        for(const auto& cell:reachable){
            QVector3D separation=center(cell)-object_center;
            separation.setZ(0.0f);
            const float distance=separation.length()/units_per_meter;
            QVector3D travel=center(cell)-center(start); travel.setZ(0.0f);
            const float travel_meters=travel.length()/units_per_meter;
            const float stand_error=already_well_placed
                ? (cell==start ? 0.0f : std::numeric_limits<float>::max())
                : std::abs(distance-preferred_meters)+0.20f*travel_meters;
            if(stand_error<best_stand_error){best_stand_error=stand_error;candidate=cell;}
        }
        // Prefer a well-supported semantic instance before small differences
        // in approach placement. This avoids selecting duplicate, low-evidence
        // TV detections merely because one happens to be closer to the grid.
        const float quality_penalty=
            0.8f*static_cast<float>(1.0-object.confidence_score)+
            0.4f/std::sqrt(static_cast<float>(std::max(1,object.observation_count)));
        const float score=best_stand_error+quality_penalty;
        if(score<best_score){best_score=score;selected=&object;goal=candidate;
            target=object_center;}
    }
    if(!class_found){emit navigationPlanStatusChanged("Object not found: "+requested_destination);return true;}
    if(!selected){emit navigationPlanStatusChanged("No confirmed object is reachable through painted cells.");return true;}
    using Cell=std::pair<int,int>;
    std::map<Cell,float> cost; std::map<Cell,Cell> parent;
    struct Node{float f;Cell c;}; struct Greater{bool operator()(const Node&a,const Node&b)const{return a.f>b.f;}};
    std::priority_queue<Node,std::vector<Node>,Greater> open; cost[start]=0; open.push({0,start});
    while(!open.empty()){
        const Cell c=open.top().c; open.pop(); if(c==goal) break;
        for(const auto& d:dirs){Cell n{c.first+d[0],c.second+d[1]};
            if(!walkable_cells_.count(n))continue;
            if(d[0]&&d[1]&&(!walkable_cells_.count({c.first+d[0],c.second})||
                             !walkable_cells_.count({c.first,c.second+d[1]})))continue;
            const float nc=cost[c]+(d[0]&&d[1]?1.4142f:1.f);
            if(!cost.count(n)||nc<cost[n]){cost[n]=nc;parent[n]=c;
                const float h=std::hypot(float(n.first-goal.first),float(n.second-goal.second));
                open.push({nc+h,n});}}
    }
    if(start!=goal&&!parent.count(goal)){emit navigationPlanStatusChanged("Painted cells are disconnected.");return true;}
    std::vector<Cell> cells; for(Cell c=goal;;c=parent[c]){cells.push_back(c);if(c==start)break;}
    std::reverse(cells.begin(),cells.end());
    std::vector<QVector3D> aligned,world; aligned.reserve(cells.size());world.reserve(cells.size());
    for(const Cell& c:cells){aligned.push_back(center(c));world.push_back(aligned_to_world.map(center(c)));}
    const QVector3D navigation_target =
        world_to_aligned.map(selected->position_world);
    viewer_->setNavigationPlan(aligned, navigation_target,
        QString("%1 #%2").arg(selected->name).arg(selected->id));
    navigation_target_center_aligned_=navigation_target; has_navigation_target_=true;
    robot_path_player_->setWorldToAlignedTransform(world_to_aligned);
    if(!robot_path_player_->setPlannedPath(world,selected->position_world)) return true;
    selectSemanticObject(selected->id); setPathEditingEnabled(false);
    constrained_z_up_navigation_=true; viewer_->setZUpGizmo(true);
    restoreNavigationView(); navigation_has_started_=true;
    robot_path_player_->togglePlayPause();
    QVector3D final_separation=aligned.back()-navigation_target;
    final_separation.setZ(0.0f);
    emit navigationPlanStatusChanged(
        QString("Painted-cell navigation: %1 #%2 | %3 cells | stop %4 m away")
            .arg(selected->name).arg(selected->id).arg(cells.size())
            .arg(final_separation.length()/units_per_meter,0,'f',2));
    return true;
}

void ViewerController::planNavigationRequest(const QString& request)
{
    viewer_->clearNavigationPlan();
    has_navigation_target_ = false;
    // A new command must not leave the previous object highlighted if relation
    // resolution or route planning fails.
    selectSemanticObject(-1);
    restoreNavigationView();
    const ParsedNavigationCommand command = navigation_intent_parser_.parse(request);
    if (!command.isValid() || command.intent != NavigationIntent::NavigateTo) {
        emit navigationPlanStatusChanged(command.error.isEmpty()
            ? "Please enter a destination command." : command.error);
        return;
    }
    const bool has_closed_free_zone =
        free_zone_closed_ && free_zone_vertices_.size() >= 3;
    if (walkable_cells_.empty() && !has_closed_free_zone &&
        navigation_path_world_.size() < 2) {
        emit navigationPlanStatusChanged("Load a robot path or walkable-cell map first.");
        return;
    }
    QString query = command.destination.trimmed().toLower();
    query.replace('_', ' '); query.replace('-', ' '); query = query.simplified();
    QString canonical = semantic_aliases_.value(query, query);
    // Be tolerant of ordinary language variants not explicitly listed in the
    // database alias table (for example "mugs" -> "mug").
    auto has_class = [this](const QString& name) {
        for (const SemanticObject& object : semantic_objects_)
            if (object.name.compare(name, Qt::CaseInsensitive) == 0) return true;
        return false;
    };
    if (!has_class(canonical) && canonical.endsWith('s') &&
        has_class(canonical.left(canonical.size() - 1)))
        canonical.chop(1);
    if (!has_class(canonical)) {
        for (const SemanticObject& object : semantic_objects_) {
            if (object.name.compare(query, Qt::CaseInsensitive) == 0) {
                canonical = object.name;
                break;
            }
        }
    }
    // Alignment may have changed after the semantic file was loaded.
    // Refresh the cached geometric relations before resolving this request.
    rebuildSpatialRelations();
    relation_target_ids_.clear();
    preferred_navigation_object_id_ = -1;
    if (!command.relation.isEmpty() && !command.reference.isEmpty()) {
        std::set<int> reference_ids;
        const QString reference = semantic_aliases_.value(
            command.reference.toLower().simplified(), command.reference.toLower().simplified());
        for (const SemanticObject& object : semantic_objects_)
            if (object.name.compare(reference, Qt::CaseInsensitive) == 0)
                reference_ids.insert(object.id);
        for (const SpatialRelation& relation : spatial_relations_)
            if (relation.type == command.relation && reference_ids.count(relation.reference_id) &&
                relation.score >= 0.25f)
                relation_target_ids_.insert(relation.subject_id);
        if (relation_target_ids_.empty()) {
            emit navigationPlanStatusChanged(QString("No %1 %2 relation was found.")
                .arg(command.destination, command.relation));
            return;
        }
    }
    // Resolve duplicate detections before considering route length. Otherwise
    // a weak duplicate near the robot can beat the well-supported instance and
    // create a zero-length route (notably for the TVs near the mug area).
    double best_semantic_quality = -std::numeric_limits<double>::infinity();
    for (const SemanticObject& object : semantic_objects_) {
        if (object.name.compare(canonical, Qt::CaseInsensitive) != 0 ||
            object.review != SemanticReviewStatus::Confirmed)
            continue;
        if (!relation_target_ids_.empty() &&
            !relation_target_ids_.count(object.id))
            continue;
        const double quality = object.confidence_score +
            0.12 * std::log1p(static_cast<double>(object.observation_count));
        if (quality > best_semantic_quality) {
            best_semantic_quality = quality;
            preferred_navigation_object_id_ = object.id;
        }
    }
    if (!walkable_cells_.empty() &&
        planThroughWalkableCells(canonical, command.destination)) return;
    if (navigation_path_world_.size() < 2 && has_closed_free_zone &&
        planThroughFreeZone(canonical, command.destination)) return;

    const auto& path = navigation_path_world_;
    std::vector<float> cumulative(path.size(), 0.0f);
    for (std::size_t i = 1; i < path.size(); ++i)
        cumulative[i] = cumulative[i - 1] + (path[i] - path[i - 1]).length();
    struct Projection { int segment = -1; float amount = 0.0f;
                        float arc = 0.0f; float offset = 0.0f; QVector3D point; };
    const auto project_to_path = [&](const QVector3D& position) {
        Projection best_projection;
        best_projection.offset = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i + 1 < path.size(); ++i) {
            const QVector3D edge = path[i + 1] - path[i];
            const float length_squared = edge.lengthSquared();
            if (length_squared < 1.0e-10f) continue;
            const float amount = std::clamp(QVector3D::dotProduct(
                position - path[i], edge) / length_squared, 0.0f, 1.0f);
            const QVector3D projected = path[i] + amount * edge;
            const float offset = (projected - position).length();
            if (offset < best_projection.offset) {
                best_projection = {static_cast<int>(i), amount,
                    cumulative[i] + amount * edge.length(), offset, projected};
            }
        }
        return best_projection;
    };
    const QVector3D current_position = navigation_has_started_
        ? robot_path_player_->currentWorldPosition() : path.front();
    const Projection start = project_to_path(current_position);

    const QMatrix4x4 transform = viewer_->sceneWorldToAlignedTransform();
    bool corridor_transform_ok = false;
    const QMatrix4x4 aligned_to_world = transform.inverted(&corridor_transform_ok);
    const float units_per_meter = std::max(
        transform.mapVector(QVector3D(1, 0, 0)).length(), 1.0e-5f);
    // Four 20 cm cells across, centered on the hand-picked path.
    const float corridor_half_width = 0.40f * units_per_meter;

    struct Candidate { const SemanticObject* object = nullptr;
                       const SemanticObservation* observation = nullptr;
                       Projection projection; QVector3D destination_world;
                       float score = 0.0f; };
    Candidate best;
    best.score = std::numeric_limits<float>::max();
    bool class_found = false;
    bool confirmed_found = false;
    for (const SemanticObject& object : semantic_objects_) {
        if (object.name.compare(canonical, Qt::CaseInsensitive) != 0) continue;
        if (preferred_navigation_object_id_ >= 0 &&
            object.id != preferred_navigation_object_id_) continue;
        if (!relation_target_ids_.empty() && !relation_target_ids_.count(object.id)) continue;
        class_found = true;
        if (object.review != SemanticReviewStatus::Confirmed) continue;
        confirmed_found = true;
        const QVector3D object_center = object.position_world;
        const QVector3D object_size = object.bounds_max_world - object.bounds_min_world;
        const float preferred_distance = std::clamp(
            0.8f + std::max(object_size.x(), object_size.y()), 0.8f, 2.5f);
        Projection projection = project_to_path(object_center);
        if (projection.segment < 0) continue;
        QVector3D destination_world = projection.point;
        const bool already_well_placed=isGoodSemanticViewpoint(
            object,transform.map(current_position),transform,units_per_meter,
            preferred_distance);
        if (already_well_placed) {
            projection=start;
            destination_world=current_position;
        } else if (corridor_transform_ok) {
            const QVector3D projected_aligned = transform.map(projection.point);
            const QVector3D object_aligned = transform.map(object_center);
            QVector3D toward_object = object_aligned - projected_aligned;
            toward_object.setZ(0.0f);
            const float lateral_distance = toward_object.length();
            const float preferred_aligned = preferred_distance * units_per_meter;
            const float approach = std::clamp(
                lateral_distance - preferred_aligned, 0.0f,
                corridor_half_width);
            if (lateral_distance > 1.0e-6f && approach > 0.0f) {
                toward_object *= approach / lateral_distance;
                QVector3D destination_aligned = projected_aligned + toward_object;
                destination_aligned.setZ(projected_aligned.z());
                destination_world = aligned_to_world.map(destination_aligned);
            }
        }
        QVector3D separation = transform.map(destination_world) -
            transform.map(object_center);
        separation.setZ(0.0f);
        const float stand_distance = separation.length() / units_per_meter;
        const float route_distance = std::abs(projection.arc - start.arc);
        const float quality_penalty =
            0.8f * static_cast<float>(1.0 - object.confidence_score) +
            0.4f / std::sqrt(static_cast<float>(
                std::max(1, object.observation_count)));
        const float score = (already_well_placed ? 0.0f :
            std::abs(stand_distance - preferred_distance)) +
            quality_penalty + 0.20f * route_distance;
        if (score < best.score)
            best = {&object, nullptr, projection, destination_world, score};
    }
    if (!class_found) {
        emit navigationPlanStatusChanged(
            QString("Object not found: %1").arg(command.destination));
        return;
    }
    if (!confirmed_found || !best.object) {
        emit navigationPlanStatusChanged(
            QString("No confirmed navigable instance of %1").arg(canonical));
        return;
    }

    std::vector<QVector3D> route_world;
    const auto append_distinct = [&](const QVector3D& point) {
        if (route_world.empty() || (route_world.back() - point).length() > 1.0e-5f)
            route_world.push_back(point);
    };
    append_distinct(start.point);
    if (start.arc <= best.projection.arc) {
        for (int vertex = start.segment + 1;
             vertex <= best.projection.segment; ++vertex)
            append_distinct(path[static_cast<std::size_t>(vertex)]);
    } else {
        for (int vertex = start.segment;
             vertex > best.projection.segment; --vertex)
            append_distinct(path[static_cast<std::size_t>(vertex)]);
    }
    append_distinct(best.projection.point);
    append_distinct(best.destination_world);
    std::vector<QVector3D> route_aligned;
    route_aligned.reserve(route_world.size());
    float distance = 0.0f;
    for (std::size_t i = 0; i < route_world.size(); ++i) {
        route_aligned.push_back(transform.map(route_world[i]));
        if (i) distance += (route_world[i] - route_world[i - 1]).length();
    }
    const QVector3D object_center_world = best.object->position_world;
    navigation_target_center_aligned_ = transform.map(object_center_world);
    has_navigation_target_ = true;
    viewer_->setNavigationPlan(route_aligned, navigation_target_center_aligned_,
        QString("%1 #%2").arg(best.object->name).arg(best.object->id));
    selectSemanticObject(best.object->id);
    robot_path_player_->setWorldToAlignedTransform(transform);
    if (!robot_path_player_->setPlannedPath(route_world, object_center_world)) {
        emit navigationPlanStatusChanged(
            "Route preview created, but playback setup failed: " +
            robot_path_player_->lastError());
        return;
    }
    setPathEditingEnabled(false);
    constrained_z_up_navigation_ = true;
    viewer_->setZUpGizmo(true);
    restoreNavigationView();
    navigation_has_started_ = true;
    robot_path_player_->togglePlayPause();
    emit navigationPlanStatusChanged(
        QString("Four-cell corridor: %1 #%2 | route %3 m | stop near %4")
            .arg(best.object->name).arg(best.object->id)
            .arg(distance, 0, 'f', 2)
            .arg(best.observation ? best.observation->image : QString()));
}

void ViewerController::reviewSemanticObject(
    int object_id, SemanticReviewStatus status)
{
    for (SemanticObject& object : semantic_objects_) {
        if (object.id == object_id) { object.review = status; break; }
    }
    semantic_reviews_dirty_ = true;
    rebuildSpatialRelations();
    viewer_->setSemanticObjects(semantic_objects_);
    emit semanticDatabaseChanged("Review status updated (in memory)", QString());
    emit semanticSelectionChanged(object_id);
}

void ViewerController::loadRobotPath(QWidget* dialog_parent)
{
    viewer_->clearNavigationPlan();
    has_navigation_target_ = false;
    arrival_view_animation_->stop();
    viewer_->setVerticalFieldOfView(45.0f);
    QFileDialog dialog(
        dialog_parent, "Load saved robot path",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data");
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::ExistingFile);
    dialog.setNameFilters({"JSON path files (*.json)", "All files (*)"});
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);
    if (dialog.exec() != QDialog::Accepted) return;
    const QStringList files = dialog.selectedFiles();
    if (files.isEmpty()) return;
    const QString path = files.constFirst();
    navigation_path_world_.clear();
    if (!robot_path_player_->loadPath(path)) {
        emit robotPlaybackStatusChanged(
            robot_path_player_->lastError(), path);
        return;
    }
    navigation_path_world_ = robot_path_player_->worldPoints();
    navigation_has_started_ = false;
    if (!navigation_path_world_.empty()) {
        free_zone_default_start_world_ = navigation_path_world_.front();
        has_free_zone_default_start_ = true;
    }
    robot_pose_initialized_ = false;
    robot_path_player_->setWorldToAlignedTransform(
        viewer_->sceneWorldToAlignedTransform());
    refreshLoadedRobotPathDisplay();
    emit robotPlaybackStatusChanged("Robot path loaded", path);
}

void ViewerController::toggleRobotPlayback()
{
    if (robot_path_player_->hasPath()) {
        if (!robot_path_player_->isPlaying()) restoreNavigationView();
        setPathEditingEnabled(false);
        constrained_z_up_navigation_ = true;
        viewer_->setZUpGizmo(true);
        robot_path_player_->setWorldToAlignedTransform(
            viewer_->sceneWorldToAlignedTransform());
    }
    robot_path_player_->togglePlayPause();
}

void ViewerController::stopRobotPlayback()
{
    robot_path_player_->stop();
}

void ViewerController::clearLoadedPath()
{
    viewer_->clearNavigationPlan();
    has_navigation_target_ = false;
    arrival_view_animation_->stop();
    viewer_->setVerticalFieldOfView(45.0f);
    robot_path_player_->clearPath();
    robot_pose_initialized_ = false;
    navigation_path_world_.clear();
    navigation_has_started_ = false;
    manual_path_.clear();
    selected_manual_path_point_ = -1;
    dragging_manual_path_point_ = false;
    viewer_->setManualPath({}, -1);
    emit manualPathStatusChanged("No manual path loaded", QString());
}

void ViewerController::setRobotPlaybackSpeed(float meters_per_second)
{
    robot_path_player_->setSpeedMetersPerSecond(meters_per_second);
}

void ViewerController::openPlyFile(QWidget* dialog_parent)
{
    QFileDialog dialog(
        dialog_parent, "Open Gaussian or point-cloud PLY", QDir::homePath());
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::ExistingFile);
    dialog.setNameFilters({"PLY files (*.ply)", "All files (*)"});
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);

    if (dialog.exec() != QDialog::Accepted) return;

    const QStringList selected_files = dialog.selectedFiles();
    if (selected_files.isEmpty()) return;
    const QString path = selected_files.constFirst();

    emit loadStatusChanged("Loading " + path + "...", path);
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();

    // Release the previous renderer scene before the loader allocates a new
    // large PLY representation, avoiding both scenes overlapping in memory.
    viewer_->setGaussianPoints({});
    scene_alignment_ready_ = false;
    const bool loaded = point_processing_.loadPly(path.toStdString());
    QApplication::restoreOverrideCursor();

    if (!loaded) {
        viewer_->setGaussianPoints({});
        emit loadStatusChanged(
            "Load failed: " + QString::fromStdString(point_processing_.lastError()),
            path);
        return;
    }

    const GaussianPlyMetadata metadata = point_processing_.metadata();
    const std::size_t splat_count = point_processing_.splatCount();
    viewer_->setGaussianPoints(
        point_processing_.points(), metadata.has_scale, metadata.has_sh_dc,
        metadata.sh_degree);
    scene_alignment_ready_ = false;
    if (constrained_z_up_navigation_) {
        // A newly loaded scene needs its own alignment and provisional start.
        setConstrainedZUpNavigation(true);
    }
    point_processing_.clear();
    const QString data_description = metadata.isComplete3DGS()
        ? QString("3DGS | SH degree %1").arg(metadata.sh_degree)
        : QString("point cloud");
    emit loadStatusChanged(
        QString("Loaded %1 points | uploaded %2 | %3")
            .arg(static_cast<qulonglong>(splat_count))
            .arg(static_cast<qulonglong>(viewer_->uploadedSplatCount()))
            .arg(data_description),
        path);
    updateTrajectoryForScene();
}

bool ViewerController::ensureSceneAlignment()
{
    if (scene_alignment_ready_) return true;
    scene_alignment_ready_ = viewer_->autoAlignSceneUp();
    emit floorAlignmentStatusChanged(scene_alignment_ready_
        ? "Scene alignment fixed to +Z for this PLY."
        : "Automatic Z-up alignment failed; free-zone coordinates may be invalid.");
    return scene_alignment_ready_;
}

void ViewerController::openTrajectoryFile(QWidget* dialog_parent)
{
    QFileDialog dialog(
        dialog_parent, "Open optimized camera poses",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data");
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setFileMode(QFileDialog::ExistingFile);
    dialog.setNameFilters({"Camera pose files (*.txt)", "All files (*)"});
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);
    if (dialog.exec() != QDialog::Accepted) return;

    const QStringList selected_files = dialog.selectedFiles();
    if (selected_files.isEmpty()) return;
    const QString path = selected_files.constFirst();
    if (!trajectory_processing_.loadOptimizedCameraPoses(path.toStdString())) {
        viewer_->setMiniMapTrajectory({}, {});
        emit trajectoryStatusChanged(
            "Trajectory load failed: " +
                QString::fromStdString(trajectory_processing_.lastError()),
            path);
        return;
    }
    updateTrajectoryForScene();
    emit trajectoryStatusChanged(
        QString("Loaded and smoothed %1 camera poses")
            .arg(static_cast<qulonglong>(trajectory_processing_.poses().size())),
        path);
}

void ViewerController::saveManualPath(QWidget* dialog_parent)
{
    if (manual_path_.empty()) {
        emit manualPathStatusChanged(
            "No manual path points to save", QString());
        return;
    }

    QFileDialog dialog(
        dialog_parent, "Save manual robot path",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data/manual_path.json");
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setNameFilters({"JSON path files (*.json)", "All files (*)"});
    dialog.setDefaultSuffix("json");
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);
    if (dialog.exec() != QDialog::Accepted) return;
    const QStringList selected_files = dialog.selectedFiles();
    if (selected_files.isEmpty()) return;
    const QString path = selected_files.constFirst();

    bool invertible = false;
    const QMatrix4x4 world_from_aligned =
        viewer_->sceneWorldToAlignedTransform().inverted(&invertible);
    if (!invertible) {
        emit manualPathStatusChanged(
            "Cannot save path: scene transform is not invertible", path);
        return;
    }

    QJsonArray points;
    for (std::size_t index = 0; index < manual_path_.size(); ++index) {
        const QVector3D world = world_from_aligned.map(manual_path_[index]);
        QJsonObject point;
        point["index"] = static_cast<int>(index);
        point["x"] = static_cast<double>(world.x());
        point["y"] = static_cast<double>(world.y());
        point["z"] = static_cast<double>(world.z());
        points.append(point);
    }

    QJsonObject root;
    root["version"] = 1;
    root["coordinate_frame"] = "3dgs_world";
    root["path_type"] = "ordered_polyline";
    root["closed"] = false;
    root["suggested_interpolation"] = "linear_arc_length";
    root["point_count"] = static_cast<int>(manual_path_.size());
    root["points"] = points;

    QSaveFile output(path);
    if (!output.open(QIODevice::WriteOnly)) {
        emit manualPathStatusChanged(
            "Cannot open path file for writing: " + output.errorString(), path);
        return;
    }
    const QByteArray json = QJsonDocument(root).toJson(QJsonDocument::Indented);
    if (output.write(json) != json.size() || !output.commit()) {
        emit manualPathStatusChanged(
            "Failed to save path: " + output.errorString(), path);
        return;
    }
    emit manualPathStatusChanged(
        QString("Saved %1 ordered path points")
            .arg(static_cast<qulonglong>(manual_path_.size())),
        path);
}

void ViewerController::updateTrajectoryForScene()
{
    if (trajectory_processing_.poses().empty()) return;

    const QMatrix4x4 qt_transform = viewer_->sceneWorldToAlignedTransform();
    Mat4d world_to_aligned;
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            world_to_aligned(row, column) = qt_transform(row, column);

    TrajectorySmoothingOptions options;
    if (!trajectory_processing_.smoothTrajectory(world_to_aligned, options)) {
        viewer_->setMiniMapTrajectory({}, {});
        emit trajectoryStatusChanged(
            "Trajectory smoothing failed: " +
                QString::fromStdString(trajectory_processing_.lastError()), {});
        return;
    }

    std::vector<QVector3D> raw_positions;
    raw_positions.reserve(trajectory_processing_.poses().size());
    for (const OptimizedCameraPose& pose : trajectory_processing_.poses()) {
        raw_positions.push_back(qt_transform.map(QVector3D(
            static_cast<float>(pose.position_world.x()),
            static_cast<float>(pose.position_world.y()),
            static_cast<float>(pose.position_world.z()))));
    }
    std::vector<QVector3D> smooth_positions;
    smooth_positions.reserve(trajectory_processing_.smoothedTrajectory().size());
    for (const TrajectoryPoint& point :
         trajectory_processing_.smoothedTrajectory()) {
        smooth_positions.emplace_back(
            static_cast<float>(point.position.x()),
            static_cast<float>(point.position.y()),
            static_cast<float>(point.position.z()));
    }
    viewer_->setMiniMapTrajectory(raw_positions, smooth_positions);
}

void ViewerController::resetView()
{
    yaw_degrees_ = 0.0f;
    pitch_degrees_ = 0.0f;
    translation_ = {};
    camera_distance_ = constrained_z_up_navigation_
        ? kNavigateInitialCameraDistance : kInitialCameraDistance;
    if (constrained_z_up_navigation_ &&
        (!navigation_path_world_.empty() || has_free_zone_default_start_)) {
        z_up_camera_position_ = viewer_->sceneWorldToAlignedTransform().map(
            !navigation_path_world_.empty() ? navigation_path_world_.front()
                                            : free_zone_default_start_world_);
        z_up_camera_position_.setZ(viewer_->robotEyeHeightAligned(
            1.0f, z_up_camera_position_.z()));
    } else {
        z_up_camera_position_ = QVector3D(0.0f, -camera_distance_, 0.0f);
    }
    applyTransform();
}

void ViewerController::setConstrainedZUpNavigation(bool enabled)
{
    if (enabled) {
        ensureSceneAlignment();
    }
    constrained_z_up_navigation_ = enabled;
    yaw_degrees_ = 0.0f;
    pitch_degrees_ = 0.0f;
    if (enabled) {
        translation_ = {};
        camera_distance_ = kNavigateInitialCameraDistance;
        const QMatrix4x4 world_to_aligned =
            viewer_->sceneWorldToAlignedTransform();
        if (robot_pose_initialized_ && robot_path_player_->hasPath()) {
            z_up_camera_position_ = world_to_aligned.map(
                robot_path_player_->currentWorldPosition());
        } else if (!navigation_path_world_.empty()) {
            z_up_camera_position_ = world_to_aligned.map(
                navigation_path_world_.front());
        } else if (has_free_zone_default_start_) {
            z_up_camera_position_ = world_to_aligned.map(
                free_zone_default_start_world_);
        } else {
            z_up_camera_position_ = QVector3D(
                0.0f, -kNavigateInitialCameraDistance, 0.0f);
        }
        z_up_camera_position_.setZ(viewer_->robotEyeHeightAligned(
            1.0f, z_up_camera_position_.z()));
    } else {
        // Explore mode returns to a complete-scene orbit view.
        translation_ = {};
        camera_distance_ = kInitialCameraDistance;
    }
    drag_mode_ = DragMode::None;
    viewer_->setZUpGizmo(enabled);
    updateTrajectoryForScene();
    robot_path_player_->setWorldToAlignedTransform(
        viewer_->sceneWorldToAlignedTransform());
    refreshLoadedRobotPathDisplay();
    applyTransform();
}

void ViewerController::setPathEditingEnabled(bool enabled)
{
    if (enabled) {
        if (ensureSceneAlignment()) {
            emit floorAlignmentStatusChanged(
                "Z-up aligned for orthographic path editing.");
        }
        constrained_z_up_navigation_ = false;
        path_editing_enabled_ = true;
        drag_mode_ = DragMode::None;
        selected_manual_path_point_ = -1;
        if (!trajectory_processing_.smoothedTrajectory().empty())
            manual_path_height_ = static_cast<float>(
                trajectory_processing_.trajectoryHeight());
        viewer_->setZUpGizmo(true);
        viewer_->setPathEditMode(true);
        updateTrajectoryForScene();
        robot_path_player_->setWorldToAlignedTransform(
            viewer_->sceneWorldToAlignedTransform());
        refreshLoadedRobotPathDisplay();
    } else {
        path_editing_enabled_ = false;
        dragging_manual_path_point_ = false;
        selected_manual_path_point_ = -1;
        viewer_->setPathEditMode(false);
    }
    updateManualPathDisplay();
}

void ViewerController::setFreeZoneEditingEnabled(bool enabled)
{
    free_zone_editing_enabled_ = enabled;
    dragging_free_zone_vertex_ = false;
    selected_free_zone_vertex_ = -1;
    if (enabled) {
        if (ensureSceneAlignment())
            emit floorAlignmentStatusChanged("Z-up aligned for free-zone editing.");
        constrained_z_up_navigation_ = false;
        path_editing_enabled_ = false;
        drag_mode_ = DragMode::None;
        free_zone_height_ = viewer_->robotEyeHeightAligned(
            0.02f, manual_path_height_);
        // A free zone represents floor area. Keep every component on one
        // detected floor plane; inherited path/camera heights must not leak
        // into polygon visualization.
        for (QVector3D& point : free_zone_vertices_)
            point.setZ(free_zone_height_);
        for (auto& polygon : completed_free_zone_polygons_)
            for (QVector3D& point : polygon) point.setZ(free_zone_height_);
        viewer_->setZUpGizmo(true);
        viewer_->setPathEditMode(true);
    } else if (!path_editing_enabled_) {
        viewer_->setPathEditMode(false);
    }
    updateFreeZoneDisplay();
}

void ViewerController::updateFreeZoneDisplay()
{
    viewer_->setFreeZone(
        completed_free_zone_polygons_, free_zone_vertices_, free_zone_closed_,
        selected_free_zone_vertex_);
    const std::size_t polygon_count = completed_free_zone_polygons_.size() +
        (free_zone_closed_ ? 1U : 0U);
    emit freeZoneStatusChanged(
        QString("%1 merged polygon(s) | %2 active vertices | %3")
            .arg(polygon_count)
            .arg(free_zone_vertices_.size())
            .arg(free_zone_closed_ ? "closed walkable zone" : "open boundary"),
        QString());
}

void ViewerController::closeFreeZone()
{
    if (free_zone_vertices_.size() < 3) {
        emit freeZoneStatusChanged("A free zone needs at least three vertices.", QString());
        return;
    }
    if (free_zone_seed_vertex_count_ > 0 &&
        free_zone_vertices_.size() <= free_zone_seed_vertex_count_) {
        emit freeZoneStatusChanged(
            "Add at least one new outer-boundary vertex before closing.", QString());
        return;
    }
    bool overlaps_existing = completed_free_zone_polygons_.empty();
    if (!completed_free_zone_polygons_.empty()) {
        QPolygonF active_polygon;
        for (const QVector3D& point : free_zone_vertices_)
            active_polygon << QPointF(point.x(), point.y());
        QPainterPath active_path;
        active_path.addPolygon(active_polygon);
        active_path.closeSubpath();
        for (const auto& vertices : completed_free_zone_polygons_) {
            QPolygonF completed_polygon;
            for (const QVector3D& point : vertices)
                completed_polygon << QPointF(point.x(), point.y());
            QPainterPath completed_path;
            completed_path.addPolygon(completed_polygon);
            completed_path.closeSubpath();
            if (!active_path.intersected(completed_path).isEmpty()) {
                overlaps_existing = true;
                break;
            }
        }
    }
    free_zone_closed_ = true;
    free_zone_seed_vertex_count_ = 0;
    selected_free_zone_vertex_ = -1;
    updateFreeZoneDisplay();
    if (!overlaps_existing)
        emit freeZoneStatusChanged(
            "Polygon saved as another component, but it does not overlap the existing free zone; navigation may remain disconnected.",
            QString());
}

void ViewerController::reopenFreeZone()
{
    free_zone_closed_ = false;
    updateFreeZoneDisplay();
}

void ViewerController::startJoinedFreeZone(
    int shared_edge_count, bool reverse_direction)
{
    if (!free_zone_closed_ || free_zone_vertices_.size() < 3) {
        emit freeZoneStatusChanged(
            "Close the current polygon before starting a joined polygon.", QString());
        return;
    }
    if (selected_free_zone_vertex_ < 0 ||
        selected_free_zone_vertex_ >= static_cast<int>(free_zone_vertices_.size())) {
        emit freeZoneStatusChanged(
            "Select a vertex on the closed polygon first.", QString());
        return;
    }
    const int vertex_count = static_cast<int>(free_zone_vertices_.size());
    shared_edge_count = std::clamp(shared_edge_count, 1, vertex_count - 1);
    std::vector<QVector3D> shared_chain;
    shared_chain.reserve(static_cast<std::size_t>(shared_edge_count + 1));
    int index = selected_free_zone_vertex_;
    for (int edge = 0; edge <= shared_edge_count; ++edge) {
        shared_chain.push_back(free_zone_vertices_[static_cast<std::size_t>(index)]);
        index = reverse_direction
            ? (index - 1 + vertex_count) % vertex_count
            : (index + 1) % vertex_count;
    }
    completed_free_zone_polygons_.push_back(free_zone_vertices_);
    free_zone_vertices_ = std::move(shared_chain);
    free_zone_closed_ = false;
    free_zone_seed_vertex_count_ = free_zone_vertices_.size();
    selected_free_zone_vertex_ = static_cast<int>(free_zone_vertices_.size()) - 1;
    updateFreeZoneDisplay();
}

void ViewerController::startAdditiveFreeZone()
{
    if (!free_zone_closed_ || free_zone_vertices_.size() < 3) {
        emit freeZoneStatusChanged(
            "Close the current polygon before adding another region.", QString());
        return;
    }
    completed_free_zone_polygons_.push_back(free_zone_vertices_);
    free_zone_vertices_.clear();
    free_zone_closed_ = false;
    free_zone_seed_vertex_count_ = 0;
    selected_free_zone_vertex_ = -1;
    updateFreeZoneDisplay();
    emit freeZoneStatusChanged(
        "Draw the next polygon anywhere; overlap the existing green area for a safe union.",
        QString());
}

void ViewerController::clearFreeZone()
{
    free_zone_vertices_.clear();
    completed_free_zone_polygons_.clear();
    free_zone_closed_ = false;
    free_zone_seed_vertex_count_ = 0;
    has_free_zone_default_start_ = false;
    selected_free_zone_vertex_ = -1;
    updateFreeZoneDisplay();
}

void ViewerController::saveFreeZone(QWidget* dialog_parent)
{
    if (!free_zone_closed_ || free_zone_vertices_.size() < 3) {
        emit freeZoneStatusChanged("Close a polygon before saving it.", QString());
        return;
    }
    if (!ensureSceneAlignment()) {
        emit freeZoneStatusChanged("Cannot save without a stable scene alignment.", QString());
        return;
    }
    QFileDialog dialog(dialog_parent, "Save walkable free zone",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data/free_zone.json");
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setNameFilter("JSON free zones (*.json)");
    dialog.setDefaultSuffix("json");
    dialog.setOption(QFileDialog::DontUseNativeDialog, true);
    if (dialog.exec() != QDialog::Accepted || dialog.selectedFiles().isEmpty()) return;
    const QString path = dialog.selectedFiles().constFirst();
    bool invertible = false;
    const QMatrix4x4 aligned_to_world =
        viewer_->sceneWorldToAlignedTransform().inverted(&invertible);
    if (!invertible) {
        emit freeZoneStatusChanged("Cannot save: scene transform is not invertible.", path);
        return;
    }
    QJsonArray polygons;
    std::vector<std::vector<QVector3D>> all_polygons = completed_free_zone_polygons_;
    all_polygons.push_back(free_zone_vertices_);
    for (std::size_t polygon_index = 0; polygon_index < all_polygons.size(); ++polygon_index) {
        QJsonArray vertices;
        for (const QVector3D& aligned : all_polygons[polygon_index]) {
            const QVector3D world = aligned_to_world.map(aligned);
            QJsonObject vertex;
            vertex["x"] = world.x(); vertex["y"] = world.y(); vertex["z"] = world.z();
            vertices.append(vertex);
        }
        QJsonObject polygon;
        polygon["id"] = static_cast<int>(polygon_index);
        polygon["vertices"] = vertices;
        polygons.append(polygon);
    }
    QJsonObject root;
    root["version"] = 1;
    root["coordinate_frame"] = "3dgs_world";
    root["type"] = "walkable_polygons";
    root["merge_operation"] = "union";
    root["robot_radius_m"] = 0.30;
    QVector3D default_start_world;
    if (!navigation_path_world_.empty()) {
        default_start_world = navigation_path_world_.front();
    } else if (has_free_zone_default_start_) {
        default_start_world = free_zone_default_start_world_;
    } else {
        QVector3D centroid_aligned;
        for (const QVector3D& point : free_zone_vertices_) centroid_aligned += point;
        centroid_aligned /= static_cast<float>(free_zone_vertices_.size());
        default_start_world = aligned_to_world.map(centroid_aligned);
    }
    QJsonArray default_start;
    default_start.append(default_start_world.x());
    default_start.append(default_start_world.y());
    default_start.append(default_start_world.z());
    root["default_start_world_m"] = default_start;
    root["polygons"] = polygons;
    free_zone_default_start_world_ = default_start_world;
    has_free_zone_default_start_ = true;
    QSaveFile output(path);
    if (!output.open(QIODevice::WriteOnly)) {
        emit freeZoneStatusChanged("Cannot save free zone: " + output.errorString(), path);
        return;
    }
    const QByteArray data = QJsonDocument(root).toJson(QJsonDocument::Indented);
    if (output.write(data) != data.size() || !output.commit()) {
        emit freeZoneStatusChanged("Failed to save free zone: " + output.errorString(), path);
        return;
    }
    emit freeZoneStatusChanged(
        QString("Saved union of %1 walkable polygons").arg(all_polygons.size()),
        path);
}

void ViewerController::loadFreeZone(QWidget* dialog_parent)
{
    const QString path = QFileDialog::getOpenFileName(dialog_parent,
        "Load walkable free zone", QStringLiteral(PROJECT_ROOT_DIR) + "/data",
        "JSON free zones (*.json);;All files (*)");
    if (path.isEmpty()) return;
    if (!ensureSceneAlignment()) {
        emit freeZoneStatusChanged("Cannot load without a stable scene alignment.", path);
        return;
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        emit freeZoneStatusChanged("Cannot open free zone: " + file.errorString(), path);
        return;
    }
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &error);
    const QJsonObject root = document.object();
    const QJsonArray polygons = root.value("polygons").toArray();
    if (error.error != QJsonParseError::NoError || !document.isObject() ||
        root.value("coordinate_frame").toString() != "3dgs_world" || polygons.isEmpty()) {
        emit freeZoneStatusChanged("Invalid 3dgs_world free-zone JSON.", path);
        return;
    }
    std::vector<std::vector<QVector3D>> loaded_polygons;
    for (const QJsonValue& polygon_value : polygons) {
        std::vector<QVector3D> loaded;
        for (const QJsonValue& value :
             polygon_value.toObject().value("vertices").toArray()) {
            const QJsonObject vertex = value.toObject();
            loaded.push_back(viewer_->sceneWorldToAlignedTransform().map(QVector3D(
                vertex.value("x").toDouble(), vertex.value("y").toDouble(),
                vertex.value("z").toDouble())));
        }
        if (loaded.size() >= 3) loaded_polygons.push_back(std::move(loaded));
    }
    if (loaded_polygons.empty()) {
        emit freeZoneStatusChanged("No valid free-zone polygons were found.", path);
        return;
    }
    free_zone_vertices_ = std::move(loaded_polygons.back());
    loaded_polygons.pop_back();
    completed_free_zone_polygons_ = std::move(loaded_polygons);
    const float loaded_fallback_height = free_zone_vertices_.front().z();
    free_zone_height_ = viewer_->robotEyeHeightAligned(
        0.02f, loaded_fallback_height);
    for (QVector3D& point : free_zone_vertices_) point.setZ(free_zone_height_);
    for (auto& polygon : completed_free_zone_polygons_)
        for (QVector3D& point : polygon) point.setZ(free_zone_height_);
    free_zone_closed_ = true;
    free_zone_seed_vertex_count_ = 0;
    has_free_zone_default_start_ = false;
    const QJsonArray saved_start = root.value("default_start_world_m").toArray();
    if (saved_start.size() == 3) {
        free_zone_default_start_world_ = QVector3D(
            saved_start[0].toDouble(), saved_start[1].toDouble(),
            saved_start[2].toDouble());
        has_free_zone_default_start_ = true;
    } else {
        // Migrate an older free-zone file from the conventional saved path
        // beside it, without requiring the user to load that path separately.
        QFile legacy_path(QFileInfo(path).absolutePath() + "/manual_path.json");
        if (legacy_path.open(QIODevice::ReadOnly)) {
            const QJsonArray points = QJsonDocument::fromJson(
                legacy_path.readAll()).object().value("points").toArray();
            if (!points.isEmpty()) {
                const QJsonObject point = points.first().toObject();
                free_zone_default_start_world_ = QVector3D(
                    point.value("x").toDouble(), point.value("y").toDouble(),
                    point.value("z").toDouble());
                has_free_zone_default_start_ = true;
            }
        }
        if (!has_free_zone_default_start_) {
            QVector3D centroid_aligned;
            for (const QVector3D& point : free_zone_vertices_) centroid_aligned += point;
            centroid_aligned /= static_cast<float>(free_zone_vertices_.size());
            bool transform_ok = false;
            const QMatrix4x4 aligned_to_world =
                viewer_->sceneWorldToAlignedTransform().inverted(&transform_ok);
            if (transform_ok) {
                free_zone_default_start_world_ = aligned_to_world.map(centroid_aligned);
                has_free_zone_default_start_ = true;
            }
        }
    }
    selected_free_zone_vertex_ = -1;
    viewer_->setFreeZone(
        completed_free_zone_polygons_, free_zone_vertices_, true, -1);
    emit freeZoneStatusChanged(
        QString("Loaded union of %1 free-zone polygons")
            .arg(completed_free_zone_polygons_.size() + 1), path);
}

void ViewerController::updateManualPathDisplay()
{
    viewer_->setManualPath(manual_path_, selected_manual_path_point_);
    emit manualPathStatusChanged(
        QString("%1 manual path points (unsaved)")
            .arg(static_cast<qulonglong>(manual_path_.size())),
        QString());
}

void ViewerController::updateWalkableCellDisplay()
{
    viewer_->setWalkableCells(
        walkable_cells_, walkable_cell_size_, walkable_floor_z_,
        walkable_grid_angle_radians_);
    const float units_per_meter = std::max(
        viewer_->sceneWorldToAlignedTransform().mapVector(
            QVector3D(1, 0, 0)).length(), 1.0e-5f);
    emit freeZoneStatusChanged(
        QString("%1 walkable cells; cell size %2 m")
            .arg(walkable_cells_.size())
            .arg(walkable_cell_size_ / units_per_meter, 0, 'f', 2), QString());
}

std::pair<int, int> ViewerController::walkableCellAt(
    const QVector3D& point) const
{
    const float cs=std::cos(walkable_grid_angle_radians_);
    const float sn=std::sin(walkable_grid_angle_radians_);
    return {static_cast<int>(std::floor((cs*point.x()+sn*point.y()) /
                                        walkable_cell_size_)),
            static_cast<int>(std::floor((-sn*point.x()+cs*point.y()) /
                                        walkable_cell_size_))};
}

void ViewerController::extendWalkableCellStroke(
    const std::pair<int, int>& destination, bool erase)
{
    const auto start = has_last_painted_walkable_cell_
        ? last_painted_walkable_cell_ : destination;
    const int dx=destination.first-start.first;
    const int dy=destination.second-start.second;
    const int steps=std::max(std::abs(dx),std::abs(dy));
    bool changed=false;
    for(int step=0;step<=steps;++step){
        const float t=steps ? static_cast<float>(step)/steps : 0.0f;
        const std::pair<int,int> cell{
            static_cast<int>(std::lround(start.first+t*dx)),
            static_cast<int>(std::lround(start.second+t*dy))};
        if(erase){changed=walkable_cells_.erase(cell)>0||changed;continue;}
        bool connected=walkable_cells_.empty();
        for(int x=-1;x<=1&&!connected;++x)
            for(int y=-1;y<=1&&!connected;++y)
                if((x||y)&&walkable_cells_.count({cell.first+x,cell.second+y}))
                    connected=true;
        if(connected) changed=walkable_cells_.insert(cell).second||changed;
    }
    last_painted_walkable_cell_=destination;
    has_last_painted_walkable_cell_=true;
    if(changed) updateWalkableCellDisplay();
}

void ViewerController::initializeWalkableCellsFromPath()
{
    if (navigation_path_world_.size() < 2) {
        emit freeZoneStatusChanged("Load a robot path before creating cells.", QString());
        return;
    }
    const QMatrix4x4 transform = viewer_->sceneWorldToAlignedTransform();
    const float units_per_meter = std::max(
        transform.mapVector(QVector3D(1, 0, 0)).length(), 1.0e-5f);
    walkable_cell_size_ = 0.12f * units_per_meter;
    const QVector3D first = transform.map(navigation_path_world_.front());
    const QVector3D second = transform.map(navigation_path_world_[1]);
    walkable_grid_angle_radians_ = std::atan2(
        second.y() - first.y(), second.x() - first.x());
    // The source path contains camera/robot poses above the floor. Using its Z
    // for the cell plane is invisible in the orthographic editor, but produces
    // a perspective parallax offset during navigation. Keep cells just above
    // the detected floor instead.
    walkable_floor_z_ = viewer_->robotEyeHeightAligned(0.02f, first.z());
    walkable_cells_.clear();
    for (std::size_t i = 0; i + 1 < navigation_path_world_.size(); ++i) {
        const QVector3D a = transform.map(navigation_path_world_[i]);
        const QVector3D b = transform.map(navigation_path_world_[i + 1]);
        const float length = QVector2D(b.x() - a.x(), b.y() - a.y()).length();
        const int samples = std::max(1, static_cast<int>(std::ceil(
            length / (walkable_cell_size_ * 0.35f))));
        for (int s = 0; s <= samples; ++s) {
            const float t = static_cast<float>(s) / samples;
            const float x = a.x() * (1-t) + b.x() * t;
            const float y = a.y() * (1-t) + b.y() * t;
            const float cs=std::cos(walkable_grid_angle_radians_);
            const float sn=std::sin(walkable_grid_angle_radians_);
            const int gx = static_cast<int>(std::floor((cs*x+sn*y) / walkable_cell_size_));
            const int gy = static_cast<int>(std::floor((-sn*x+cs*y) / walkable_cell_size_));
            // A four-cell-wide connected backbone.
            for (int dx = -1; dx <= 2; ++dx)
                for (int dy = -1; dy <= 2; ++dy)
                    walkable_cells_.insert({gx + dx, gy + dy});
        }
    }
    updateWalkableCellDisplay();
}

void ViewerController::setWalkableCellEditingEnabled(bool enabled)
{
    walkable_cell_editing_enabled_ = enabled;
    painting_walkable_cells_ = false;
    if (enabled) {
        setPathEditingEnabled(false);
        setFreeZoneEditingEnabled(false);
        constrained_z_up_navigation_ = false;
        viewer_->setZUpGizmo(true);
        viewer_->setPathEditMode(true);
        if (walkable_cells_.empty()) initializeWalkableCellsFromPath();
    } else {
        viewer_->setPathEditMode(false);
    }
}

void ViewerController::clearWalkableCells()
{
    walkable_cells_.clear();
    updateWalkableCellDisplay();
}

void ViewerController::setWalkableCellsVisible(bool visible)
{
    viewer_->setWalkableCellsVisible(visible);
}

void ViewerController::saveWalkableCells(QWidget* parent)
{
    if (walkable_cells_.empty()) return;
    QString path = QFileDialog::getSaveFileName(parent, "Save walkable cells",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data/walkable_cells.json", "JSON (*.json)");
    if (path.isEmpty()) return;
    QJsonObject root; root["version"] = 1;
    root["coordinate_frame"] = "scene_z_up_aligned";
    root["cell_size_aligned"] = walkable_cell_size_;
    root["cell_size_m"] = 0.12;
    root["floor_z_aligned"] = walkable_floor_z_;
    root["grid_angle_radians"] = walkable_grid_angle_radians_;
    QJsonArray cells;
    for (const auto& cell : walkable_cells_) {
        QJsonArray value; value.append(cell.first); value.append(cell.second);
        cells.append(value);
    }
    root["walkable_cells"] = cells;
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) ||
        file.write(QJsonDocument(root).toJson(QJsonDocument::Indented)) < 0 ||
        !file.commit()) {
        emit freeZoneStatusChanged("Could not save walkable cells.", QString()); return;
    }
    emit freeZoneStatusChanged(QString("Saved %1 walkable cells").arg(cells.size()), path);
}

void ViewerController::loadWalkableCells(QWidget* parent)
{
    const QString path = QFileDialog::getOpenFileName(parent, "Load walkable cells",
        QStringLiteral(PROJECT_ROOT_DIR) + "/data", "JSON (*.json)");
    if (path.isEmpty()) return;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    const QJsonObject root = doc.object();
    if (root.value("coordinate_frame").toString() != "scene_z_up_aligned") {
        emit freeZoneStatusChanged("Walkable-cell file is not Z-up aligned.", path); return;
    }
    std::set<std::pair<int,int>> loaded;
    for (const QJsonValue& value : root.value("walkable_cells").toArray()) {
        const QJsonArray c = value.toArray();
        if (c.size() == 2) loaded.insert({c[0].toInt(), c[1].toInt()});
    }
    if (loaded.empty()) return;
    walkable_cells_ = std::move(loaded);
    walkable_cell_size_ = root.value("cell_size_aligned").toDouble(0.12);
    const float saved_floor_z =
        root.value("floor_z_aligned").toDouble();
    // Version-1 files may contain the old camera-height plane. Re-anchor them
    // to the current scene floor when floor estimation is available.
    walkable_floor_z_ =
        viewer_->robotEyeHeightAligned(0.02f, saved_floor_z);
    walkable_grid_angle_radians_ = root.value("grid_angle_radians").toDouble();
    updateWalkableCellDisplay();
}

void ViewerController::refreshLoadedRobotPathDisplay()
{
    if (!robot_path_player_->hasPath()) return;
    manual_path_.clear();
    manual_path_.reserve(robot_path_player_->worldPoints().size());
    const QMatrix4x4 transform = viewer_->sceneWorldToAlignedTransform();
    for (const QVector3D& world : robot_path_player_->worldPoints())
        manual_path_.push_back(transform.map(world));
    selected_manual_path_point_ = -1;
    viewer_->setManualPath(manual_path_, selected_manual_path_point_);
}

void ViewerController::applyTransform()
{
    QMatrix4x4 transform;
    if (constrained_z_up_navigation_) {
        const float yaw_radians =
            qDegreesToRadians(yaw_degrees_);
        const float pitch_radians =
            qDegreesToRadians(pitch_degrees_);
        const float cos_pitch = std::cos(pitch_radians);
        const QVector3D forward(
            std::sin(yaw_radians) * cos_pitch,
            std::cos(yaw_radians) * cos_pitch,
            -std::sin(pitch_radians));
        const QVector3D world_up(0.0f, 0.0f, 1.0f);
        const QVector3D right =
            QVector3D::crossProduct(forward, world_up).normalized();
        const QVector3D camera_up =
            QVector3D::crossProduct(right, forward).normalized();

        QMatrix4x4 desired_view;
        desired_view.lookAt(
            z_up_camera_position_, z_up_camera_position_ + forward, camera_up);

        // OpenGLWidget owns a fixed view at (0,0,3). Pre-cancel it so the
        // resulting model-view matrix is the first-person Z-up camera above.
        QMatrix4x4 inverse_fixed_view;
        inverse_fixed_view.translate(0.0f, 0.0f, kInitialCameraDistance);
        transform = inverse_fixed_view * desired_view;
        viewer_->setInteractionTransform(
            transform, yaw_degrees_, pitch_degrees_);
        viewer_->setNavigationPose(z_up_camera_position_, yaw_degrees_);
        emit orientationChanged(yaw_degrees_, pitch_degrees_);
        return;
    }

    QVector3D model_offset = translation_;
    // The render camera stays at Z=3. Moving the normalized cloud toward it
    // is equivalent to dollying the camera toward the cloud.
    model_offset.setZ(kInitialCameraDistance - camera_distance_);
    transform.translate(model_offset);
    transform.rotate(pitch_degrees_, 1.0f, 0.0f, 0.0f);
    transform.rotate(yaw_degrees_, 0.0f, 1.0f, 0.0f);
    viewer_->setInteractionTransform(transform, yaw_degrees_, pitch_degrees_);
    emit orientationChanged(yaw_degrees_, pitch_degrees_);
}

void ViewerController::snapToAxis(int axis)
{
    switch (static_cast<OrientationGizmo::Axis>(axis)) {
    case OrientationGizmo::Axis::X:
        yaw_degrees_ = 90.0f;
        pitch_degrees_ = 0.0f;
        break;
    case OrientationGizmo::Axis::Y:
        yaw_degrees_ = 0.0f;
        pitch_degrees_ = -90.0f;
        break;
    case OrientationGizmo::Axis::Z:
        yaw_degrees_ = 0.0f;
        pitch_degrees_ = 0.0f;
        break;
    case OrientationGizmo::Axis::None:
        return;
    }
    applyTransform();
}

bool ViewerController::eventFilter(QObject* watched, QEvent* event)
{
    if (watched != viewer_) return QObject::eventFilter(watched, event);

    // During active playback the generated robot pose owns the camera. Pause
    // restores the normal Navigate controls; resuming emits the stored path
    // pose again and discards temporary inspection movement.
    if (robot_path_player_->isPlaying()) {
        switch (event->type()) {
        case QEvent::KeyPress:
        case QEvent::MouseButtonPress:
        case QEvent::MouseButtonRelease:
        case QEvent::MouseMove:
        case QEvent::Wheel:
            event->accept();
            return true;
        default:
            break;
        }
    }

    switch (event->type()) {
    case QEvent::KeyPress: {
        auto* key_event = static_cast<QKeyEvent*>(event);
        if (path_editing_enabled_ &&
            (key_event->key() == Qt::Key_Delete ||
             key_event->key() == Qt::Key_Backspace) &&
            selected_manual_path_point_ >= 0 &&
            selected_manual_path_point_ <
                static_cast<int>(manual_path_.size())) {
            manual_path_.erase(
                manual_path_.begin() + selected_manual_path_point_);
            selected_manual_path_point_ = -1;
            updateManualPathDisplay();
            key_event->accept();
            return true;
        }
        if (free_zone_editing_enabled_ &&
            (key_event->key() == Qt::Key_Delete ||
             key_event->key() == Qt::Key_Backspace) &&
            selected_free_zone_vertex_ >= 0 &&
            selected_free_zone_vertex_ < static_cast<int>(free_zone_vertices_.size())) {
            if (static_cast<std::size_t>(selected_free_zone_vertex_) <
                free_zone_seed_vertex_count_) {
                emit freeZoneStatusChanged(
                    "Shared-edge vertices are locked while drawing the joined polygon.",
                    QString());
                key_event->accept();
                return true;
            }
            free_zone_vertices_.erase(
                free_zone_vertices_.begin() + selected_free_zone_vertex_);
            selected_free_zone_vertex_ = -1;
            if (free_zone_vertices_.size() < 3) free_zone_closed_ = false;
            updateFreeZoneDisplay();
            key_event->accept();
            return true;
        }
        if (constrained_z_up_navigation_ &&
            (key_event->key() == Qt::Key_Up ||
             key_event->key() == Qt::Key_Down)) {
            const float vertical_step = std::max(
                camera_distance_, kMinimumCameraDistance) *
                kVerticalMoveFraction;
            z_up_camera_position_.setZ(
                z_up_camera_position_.z() +
                (key_event->key() == Qt::Key_Up
                    ? vertical_step
                    : -vertical_step));
            applyTransform();
            key_event->accept();
            return true;
        }
        break;
    }
    case QEvent::MouseButtonPress: {
        auto* mouse_event = static_cast<QMouseEvent*>(event);
        if (walkable_cell_editing_enabled_ &&
            (mouse_event->button() == Qt::LeftButton ||
             mouse_event->button() == Qt::RightButton)) {
            const auto point = viewer_->screenToPathPlane(
                mouse_event->pos(), walkable_floor_z_);
            if (point) {
                has_last_painted_walkable_cell_ = false;
                erasing_walkable_cells_ = mouse_event->button() == Qt::RightButton;
                extendWalkableCellStroke(walkableCellAt(*point), erasing_walkable_cells_);
            }
            painting_walkable_cells_ = true;
            mouse_event->accept(); return true;
        }
        if (path_editing_enabled_) {
            const int hit = viewer_->hitTestManualPathPoint(mouse_event->pos());
            if (mouse_event->button() == Qt::RightButton) {
                if (hit >= 0) {
                    manual_path_.erase(manual_path_.begin() + hit);
                    selected_manual_path_point_ = -1;
                    updateManualPathDisplay();
                }
                mouse_event->accept();
                return true;
            }
            if (mouse_event->button() == Qt::LeftButton) {
                if (hit >= 0) {
                    selected_manual_path_point_ = hit;
                    dragging_manual_path_point_ = true;
                } else {
                    const auto point = viewer_->screenToPathPlane(
                        mouse_event->pos(), manual_path_height_);
                    if (point) {
                        manual_path_.push_back(*point);
                        selected_manual_path_point_ =
                            static_cast<int>(manual_path_.size()) - 1;
                    }
                }
                updateManualPathDisplay();
                mouse_event->accept();
                return true;
            }
        }
        if (free_zone_editing_enabled_) {
            const int hit = viewer_->hitTestFreeZoneVertex(mouse_event->pos());
            if (mouse_event->button() == Qt::RightButton) {
                if (hit >= 0) {
                    if (static_cast<std::size_t>(hit) < free_zone_seed_vertex_count_) {
                        emit freeZoneStatusChanged(
                            "Shared-edge vertices are locked while drawing the joined polygon.",
                            QString());
                        mouse_event->accept();
                        return true;
                    }
                    free_zone_vertices_.erase(free_zone_vertices_.begin() + hit);
                    selected_free_zone_vertex_ = -1;
                    if (free_zone_vertices_.size() < 3) free_zone_closed_ = false;
                    updateFreeZoneDisplay();
                }
                mouse_event->accept();
                return true;
            }
            if (mouse_event->button() == Qt::LeftButton) {
                if (hit >= 0) {
                    selected_free_zone_vertex_ = hit;
                    dragging_free_zone_vertex_ =
                        static_cast<std::size_t>(hit) >= free_zone_seed_vertex_count_;
                } else if (!free_zone_closed_) {
                    const auto point = viewer_->screenToPathPlane(
                        mouse_event->pos(), free_zone_height_);
                    if (point) {
                        free_zone_vertices_.push_back(*point);
                        selected_free_zone_vertex_ =
                            static_cast<int>(free_zone_vertices_.size()) - 1;
                    }
                }
                updateFreeZoneDisplay();
                mouse_event->accept();
                return true;
            }
        }
        if (mouse_event->button() == Qt::LeftButton) {
            const auto axis = OrientationGizmo::hitTest(
                mouse_event->pos(), viewer_->size(), yaw_degrees_, pitch_degrees_,
                constrained_z_up_navigation_);
            if (axis != OrientationGizmo::Axis::None) {
                snapToAxis(static_cast<int>(axis));
                mouse_event->accept();
                return true;
            }
            last_mouse_position_ = mouse_event->pos();
            drag_start_position_ = mouse_event->pos();
            if (semantic_verification_mode_) {
                // Semantic inspection is a camera free-look, never a model
                // orbit.  The slower response makes precise box inspection
                // practical even when the object is close to the camera.
                constrained_z_up_navigation_ = true;
                drag_mode_ = DragMode::SemanticLook;
            } else if (mouse_event->modifiers() & Qt::ShiftModifier) {
                drag_mode_ = DragMode::RotatePitchOnly;
            } else {
                drag_mode_ = constrained_z_up_navigation_
                    ? DragMode::RotatePending
                    : DragMode::RotateFree;
            }
            mouse_event->accept();
            return true;
        }
        if (mouse_event->button() == Qt::RightButton) {
            last_mouse_position_ = mouse_event->pos();
            drag_mode_ = DragMode::Pan;
            mouse_event->accept();
            return true;
        }
        break;
    }
    case QEvent::MouseMove: {
        auto* mouse_event = static_cast<QMouseEvent*>(event);
        if (walkable_cell_editing_enabled_ && painting_walkable_cells_ &&
            (mouse_event->buttons() & (Qt::LeftButton | Qt::RightButton))) {
            const auto point = viewer_->screenToPathPlane(
                mouse_event->pos(), walkable_floor_z_);
            if (point) {
                extendWalkableCellStroke(
                    walkableCellAt(*point), erasing_walkable_cells_);
            }
            mouse_event->accept(); return true;
        }
        if (path_editing_enabled_ && dragging_manual_path_point_ &&
            (mouse_event->buttons() & Qt::LeftButton)) {
            const auto point = viewer_->screenToPathPlane(
                mouse_event->pos(), manual_path_height_);
            if (point && selected_manual_path_point_ >= 0 &&
                selected_manual_path_point_ <
                    static_cast<int>(manual_path_.size())) {
                manual_path_[selected_manual_path_point_] = *point;
                updateManualPathDisplay();
            }
            mouse_event->accept();
            return true;
        }
        if (free_zone_editing_enabled_ && dragging_free_zone_vertex_ &&
            (mouse_event->buttons() & Qt::LeftButton)) {
            const auto point = viewer_->screenToPathPlane(
                mouse_event->pos(), free_zone_height_);
            if (point && selected_free_zone_vertex_ >= 0 &&
                selected_free_zone_vertex_ < static_cast<int>(free_zone_vertices_.size())) {
                free_zone_vertices_[selected_free_zone_vertex_] = *point;
                updateFreeZoneDisplay();
            }
            mouse_event->accept();
            return true;
        }
        if ((drag_mode_ == DragMode::RotateFree ||
             drag_mode_ == DragMode::RotatePending ||
             drag_mode_ == DragMode::RotateYawOnly ||
             drag_mode_ == DragMode::RotatePitchOnly ||
             drag_mode_ == DragMode::SemanticLook) &&
            (mouse_event->buttons() & Qt::LeftButton)) {
            QPoint delta = mouse_event->pos() - last_mouse_position_;
            if (drag_mode_ == DragMode::RotatePending) {
                const QPoint total_delta =
                    mouse_event->pos() - drag_start_position_;
                const float horizontal =
                    static_cast<float>(std::abs(total_delta.x()));
                const float vertical =
                    static_cast<float>(std::abs(total_delta.y()));
                if (std::max(horizontal, vertical) <
                    kConstrainedDragThresholdPixels) {
                    mouse_event->accept();
                    return true;
                }
                // Select exactly one axis for the lifetime of this drag.
                // Later diagonal motion cannot introduce roll or alter the
                // other angle.
                drag_mode_ = horizontal >= vertical
                    ? DragMode::RotateYawOnly
                    : DragMode::RotatePitchOnly;
                delta = total_delta;
            }
            last_mouse_position_ = mouse_event->pos();
            if (drag_mode_ == DragMode::SemanticLook) {
                yaw_degrees_ = std::fmod(
                    yaw_degrees_ + delta.x() * kSemanticRotationDegreesPerPixel,
                    360.0f);
                pitch_degrees_ = std::clamp(
                    pitch_degrees_ + delta.y() * kSemanticRotationDegreesPerPixel,
                    -85.0f, 85.0f);
            } else if (drag_mode_ == DragMode::RotateFree ||
                drag_mode_ == DragMode::RotateYawOnly) {
                const float sensitivity = constrained_z_up_navigation_
                    ? kConstrainedRotationDegreesPerPixel
                    : kRotationDegreesPerPixel;
                yaw_degrees_ = std::fmod(
                    yaw_degrees_ + delta.x() * sensitivity,
                    360.0f);
            }
            if (drag_mode_ != DragMode::SemanticLook &&
                (drag_mode_ == DragMode::RotateFree ||
                 drag_mode_ == DragMode::RotatePitchOnly)) {
                const float sensitivity = constrained_z_up_navigation_
                    ? kConstrainedRotationDegreesPerPixel
                    : kRotationDegreesPerPixel;
                pitch_degrees_ = std::clamp(
                    pitch_degrees_ + delta.y() * sensitivity,
                    -89.0f, 89.0f);
            }
            applyTransform();
            mouse_event->accept();
            return true;
        }
        if (drag_mode_ == DragMode::Pan &&
            (mouse_event->buttons() & Qt::RightButton)) {
            const QPoint delta = mouse_event->pos() - last_mouse_position_;
            last_mouse_position_ = mouse_event->pos();
            const float visible_world_height =
                2.0f * std::tan(kHalfVerticalFieldOfViewRadians) *
                std::max(camera_distance_, kMinimumCameraDistance);
            const float world_units_per_pixel = kPanSensitivity *
                visible_world_height / std::max(1, viewer_->height());
            if (constrained_z_up_navigation_) {
                const float yaw_radians = qDegreesToRadians(yaw_degrees_);
                const QVector3D right(
                    std::cos(yaw_radians), -std::sin(yaw_radians), 0.0f);
                const QVector3D ground_forward(
                    std::sin(yaw_radians), std::cos(yaw_radians), 0.0f);
                z_up_camera_position_ +=
                    -right * (delta.x() * world_units_per_pixel) +
                    ground_forward * (delta.y() * world_units_per_pixel);
            } else {
                translation_ += QVector3D(
                    delta.x() * world_units_per_pixel,
                    -delta.y() * world_units_per_pixel,
                    0.0f);
            }
            applyTransform();
            mouse_event->accept();
            return true;
        }
        break;
    }
    case QEvent::MouseButtonRelease: {
        auto* mouse_event = static_cast<QMouseEvent*>(event);
        if (walkable_cell_editing_enabled_) {
            painting_walkable_cells_ = false;
            has_last_painted_walkable_cell_ = false;
            mouse_event->accept(); return true;
        }
        if (path_editing_enabled_ && mouse_event->button() == Qt::LeftButton) {
            dragging_manual_path_point_ = false;
            mouse_event->accept();
            return true;
        }
        if (free_zone_editing_enabled_ && mouse_event->button() == Qt::LeftButton) {
            dragging_free_zone_vertex_ = false;
            mouse_event->accept();
            return true;
        }
        if ((mouse_event->button() == Qt::LeftButton &&
             (drag_mode_ == DragMode::RotateFree ||
              drag_mode_ == DragMode::RotatePending ||
              drag_mode_ == DragMode::RotateYawOnly ||
              drag_mode_ == DragMode::RotatePitchOnly ||
              drag_mode_ == DragMode::SemanticLook)) ||
            (mouse_event->button() == Qt::RightButton &&
             drag_mode_ == DragMode::Pan)) {
            const bool finished_rotation = drag_mode_ != DragMode::Pan;
            drag_mode_ = DragMode::None;
            if (finished_rotation) viewer_->finalizeInteractionSort();
            mouse_event->accept();
            return true;
        }
        break;
    }
    case QEvent::Wheel: {
        auto* wheel_event = static_cast<QWheelEvent*>(event);
        const int angle_delta = wheel_event->angleDelta().y();
        const int pixel_delta = wheel_event->pixelDelta().y();
        const float wheel_steps = angle_delta != 0
            ? static_cast<float>(angle_delta) / 120.0f
            : static_cast<float>(pixel_delta) / 60.0f;
        if (std::abs(wheel_steps) < 1.0e-4f) break;
        if (path_editing_enabled_ || free_zone_editing_enabled_ ||
            walkable_cell_editing_enabled_) {
            viewer_->zoomPathEditView(wheel_steps);
            wheel_event->accept();
            return true;
        }
        const float previous_distance = camera_distance_;
        camera_distance_ = std::clamp(
            camera_distance_ * std::pow(
                kDollyFactorPerWheelStep, wheel_steps),
            kMinimumCameraDistance, kMaximumCameraDistance);
        if (constrained_z_up_navigation_) {
            const float yaw_radians = qDegreesToRadians(yaw_degrees_);
            const QVector3D ground_forward(
                std::sin(yaw_radians), std::cos(yaw_radians), 0.0f);
            z_up_camera_position_ +=
                ground_forward * (previous_distance - camera_distance_);
        }
        applyTransform();
        wheel_event->accept();
        return true;
    }
    default:
        break;
    }

    return QObject::eventFilter(watched, event);
}
