#include "ViewerController.h"
#include "OpenGLWidget.h"
#include "RobotPathPlayer.h"
#include "SemanticQuery.h"

#include <QVector2D>
#include <QtMath>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <vector>

// Semantic destination resolution and grid/corridor route planning.
namespace
{
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
        if(observation_offset.length()/units_per_meter<=0.50f)
            return true;
    }
    return false;
}
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
        bool valid = free_area.contains(p) &&
            viewer_->isAlignedPositionInsideScene(QVector3D(p.x(), p.y(), free_zone_height_));
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
    // Continue from the previous trip, otherwise prefer the loaded path start.
    if (navigation_has_started_ && robot_path_player_->hasPath())
        current_aligned = world_to_aligned.map(robot_path_player_->currentWorldPosition());
    else if (!navigation_path_world_.empty())
        current_aligned = world_to_aligned.map(navigation_path_world_.front());
    else if (has_free_zone_default_start_)
        current_aligned = world_to_aligned.map(free_zone_default_start_world_);
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
    navigation_has_started_ = true;
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
    // Painted cells express user intent, but the orthographic overlay can
    // visually cover sofa/table geometry. Intersect them with the 3DGS
    // obstacle map so the route cannot enter occupied or clearance cells.
    std::set<std::pair<int,int>> safe_cells;
    for (const auto& cell : walkable_cells_)
        if (viewer_->isAlignedPositionInsideScene(center(cell)))
            safe_cells.insert(cell);
    std::size_t excluded_cell_count = 0;
    if (viewer_->hasSceneWalkability()) {
        safe_cells.clear();
        for (const auto& cell : walkable_cells_) {
            if (viewer_->isAlignedPositionInsideScene(center(cell)) &&
                viewer_->isAlignedPositionWalkable(center(cell)))
                safe_cells.insert(cell);
            else
                ++excluded_cell_count;
        }
        if (safe_cells.empty()) {
            emit navigationPlanStatusChanged(
                "All painted cells overlap 3DGS obstacles or clearance zones.");
            return true;
        }
    }
    QVector3D start_position;
    if (!navigation_has_started_ && !navigation_path_world_.empty())
        start_position = world_to_aligned.map(navigation_path_world_.front());
    else if (!navigation_has_started_ && has_free_zone_default_start_)
        start_position = world_to_aligned.map(free_zone_default_start_world_);
    else
        start_position = world_to_aligned.map(robot_path_player_->currentWorldPosition());
    const auto start_result = nearest(start_position, safe_cells);
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
            if (safe_cells.count(n) && reachable.insert(n).second) flood.push(n);
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
            if(!safe_cells.count(n))continue;
            if(d[0]&&d[1]&&(!safe_cells.count({c.first+d[0],c.second})||
                             !safe_cells.count({c.first,c.second+d[1]})))continue;
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
        QString("Painted-cell navigation: %1 #%2 | %3 route cells | "
                "%4 obstacle cells excluded | stop %5 m away")
            .arg(selected->name).arg(selected->id).arg(cells.size())
            .arg(excluded_cell_count)
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
    // Validate the requested start before any planner can snap it to a route.
    if (navigation_has_started_ || !navigation_path_world_.empty() ||
        has_free_zone_default_start_) {
        const QVector3D start_world = navigation_has_started_
            ? robot_path_player_->currentWorldPosition()
            : (!navigation_path_world_.empty() ? navigation_path_world_.front()
                                               : free_zone_default_start_world_);
        if (!viewer_->isAlignedPositionInsideScene(
                viewer_->sceneWorldToAlignedTransform().map(start_world))) {
            emit navigationPlanStatusChanged(
                "Navigation start is outside the scene. Load a path starting inside the scene.");
            return;
        }
    }
    QStringList classes;
    for (const SemanticObject& object : semantic_objects_)
        if (!classes.contains(object.name)) classes.append(object.name);
    const SemanticQuery query = resolveSemanticQuery(
        command.destination, classes, semantic_aliases_);
    const QString canonical = query.canonical;
    if (canonical.isEmpty()) {
        emit navigationPlanStatusChanged("Object class not found: " + command.destination);
        return;
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
    // Intersect description matches with relation matches before ranking.
    // An empty result must stop here, because an empty ID set means no filter.
    if (!query.description_words.isEmpty()) {
        std::set<int> matches;
        for (const SemanticObject& object : semantic_objects_) {
            if (object.name.compare(canonical, Qt::CaseInsensitive) != 0 ||
                object.review != SemanticReviewStatus::Confirmed)
                continue;
            if (!relation_target_ids_.empty() && !relation_target_ids_.count(object.id))
                continue;
            if (semanticDescriptionMatches(object.description, query.description_words))
                matches.insert(object.id);
        }
        if (matches.empty()) {
            emit navigationPlanStatusChanged(
                "No confirmed object matches the description: " + command.destination);
            return;
        }
        relation_target_ids_ = std::move(matches);
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
