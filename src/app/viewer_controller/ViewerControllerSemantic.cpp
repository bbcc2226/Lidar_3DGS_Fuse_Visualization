#include "ViewerController.h"
#include "OpenGLWidget.h"
#include "SemanticQuery.h"

#include <QDateTime>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QWidget>
#include <QtMath>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

// Semantic database I/O, review state, object selection, and verification view.
namespace
{
QString semanticReviewStatusName(SemanticReviewStatus status)
{
    switch (status) {
    case SemanticReviewStatus::Confirmed: return QStringLiteral("confirmed");
    case SemanticReviewStatus::Uncertain: return QStringLiteral("uncertain");
    case SemanticReviewStatus::Incorrect: return QStringLiteral("incorrect");
    default: return QStringLiteral("unverified");
    }
}
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
        const QString review_status = json.value("review_status").toString();
        if (review_status == "confirmed") object.review = SemanticReviewStatus::Confirmed;
        else if (review_status == "uncertain") object.review = SemanticReviewStatus::Uncertain;
        else if (review_status == "incorrect") object.review = SemanticReviewStatus::Incorrect;
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

    if (QFileInfo(path).absoluteFilePath() == QFileInfo(semantic_database_path_).absoluteFilePath()) {
        emit semanticDatabaseChanged(
            "Use Update current object file to save verification into the database", QString());
        return;
    }
    QJsonArray reviews;
    int reviewed_count = 0;
    for (const SemanticObject& object : semantic_objects_) {
        QJsonObject review;
        review["object_id"] = object.id;
        review["canonical_class"] = object.name;
        review["status"] = semanticReviewStatusName(object.review);
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

void ViewerController::saveSemanticObjectFile(QWidget* dialog_parent)
{
    QFile input(semantic_database_path_);
    if (semantic_objects_.empty() || !input.open(QIODevice::ReadOnly)) {
        emit semanticDatabaseChanged("Cannot read current object file: " + input.errorString(), QString());
        return;
    }
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(input.readAll(), &error);
    input.close();
    if (error.error != QJsonParseError::NoError || !document.isObject() ||
        !document.object().value("objects").isArray()) {
        emit semanticDatabaseChanged("Current object file is invalid; no changes saved", QString());
        return;
    }
    // Preserve all source metadata and object fields, changing verification only.
    QJsonObject root = document.object();
    QJsonArray objects = root.value("objects").toArray();
    QHash<int, QString> reviews;
    for (const SemanticObject& object : semantic_objects_)
        reviews.insert(object.id, semanticReviewStatusName(object.review));
    for (int index = 0; index < objects.size(); ++index) {
        QJsonObject object = objects[index].toObject();
        const int id = object.value("object_id").toInt(-1);
        if (!reviews.contains(id)) continue;
        object["review_status"] = reviews.take(id);
        objects[index] = object;
    }
    if (!reviews.isEmpty()) {
        emit semanticDatabaseChanged(
            "Current object file has changed; reload it before saving verification", QString());
        return;
    }
    root["objects"] = objects;
    QSaveFile output(semantic_database_path_);
    if (!output.open(QIODevice::WriteOnly)) {
        emit semanticDatabaseChanged("Cannot update object file: " + output.errorString(), QString());
        return;
    }
    const QByteArray data = QJsonDocument(root).toJson(QJsonDocument::Indented);
    if (output.write(data) != data.size() || !output.commit()) {
        emit semanticDatabaseChanged("Failed to update object file: " + output.errorString(), QString());
        return;
    }
    // Reviews are loaded after the database, so update the companion file too.
    semantic_review_path_ = QFileInfo(semantic_database_path_).absolutePath() + "/" +
        QFileInfo(semantic_database_path_).completeBaseName() + ".reviews.json";
    semantic_reviews_dirty_ = true;
    saveSemanticReviews(dialog_parent);
    if (semantic_reviews_dirty_) return; // Keep the review-save error visible.
    emit semanticDatabaseChanged(
        "Updated verification in " + semantic_database_path_ +
        " and " + semantic_review_path_, QString());
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

void ViewerController::selectSemanticObject(int object_id, bool focus_view)
{
    selected_semantic_object_id_ = object_id;
    viewer_->setSelectedSemanticObject(object_id);
    emit semanticSelectionChanged(object_id);
    if (!semantic_verification_mode_ || !focus_view) return;
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
