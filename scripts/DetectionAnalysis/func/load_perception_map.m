function [lid_det_map] = load_perception_map(lid_det, ego_pose, urdf_tree)
% LOAD_PERCEPTION_MAP  Express relative detections in the vehicle-state frame.
%
%   Map coordinates are stored in x_map, y_map and z_map. Relative fields
%   are preserved. If 3D boxes exist, their map vertices are in bbox_map.
%
%   The ego pose describes urdf_tree.RootFrame, unless a child_frame_id
%   is supplied in ego_pose. Its frame_id defines the output map frame.
%   Pose angles and detection orientations are in degrees. Timestamps must
%   use the same clock and units (seconds). Each message uses the nearest
%   finite ego pose within 0.1 s. Unmatched map data remains NaN.
%   Map orientations are roll_map, pitch_map and orientation_map (yaw).
%   ego_idx and pose_dt report the matched pose index and time difference.


%  Initialise map fields

lid_det_map = lid_det;
lid_det_map.frame_id_map = ego_pose.frame_id;

lid_det_map.x_map           = nan(size(lid_det.x_cog));
lid_det_map.y_map           = nan(size(lid_det.y_cog));
lid_det_map.z_map           = nan(size(lid_det.z_cog));
lid_det_map.roll_map        = nan(size(lid_det.orientation));
lid_det_map.pitch_map       = nan(size(lid_det.orientation));
lid_det_map.orientation_map = nan(size(lid_det.orientation));

n_msgs = size(lid_det.x_cog,1);
lid_det_map.ego_idx = nan(n_msgs,1);
lid_det_map.pose_dt = nan(n_msgs,1);

has_bbox = isfield(lid_det,'bbox') && ...
           all(isfield(lid_det.bbox,{'x','y','z'}));
if has_bbox
    lid_det_map.bbox_map = lid_det.bbox;
    lid_det_map.bbox_map.x(:) = NaN;
    lid_det_map.bbox_map.y(:) = NaN;
    lid_det_map.bbox_map.z(:) = NaN;
end

if ~any(isfinite(lid_det.x_cog(:)))
    return;
end

%  Match timestamps (one timestamp per message or per detection)

max_pose_dt = 0.1;
if isvector(lid_det.stamp) && numel(lid_det.stamp) == n_msgs
    det_stamp = lid_det.stamp(:);
elseif isequal(size(lid_det.stamp),size(lid_det.x_cog))
    det_stamp = max(lid_det.stamp,[],2,'omitnan');
else
    error('load_perception_map:InvalidStamps', ...
        'Detection timestamps must be per-message or match the detection arrays.');
end

valid_ego = find(all(isfinite([ego_pose.stamp(:), ...
    ego_pose.x(:),ego_pose.y(:),ego_pose.z(:), ...
    ego_pose.roll(:),ego_pose.pitch(:),ego_pose.yaw(:)]),2));
messages = find(isfinite(det_stamp));
if isempty(valid_ego) || isempty(messages)
    return;
end

% Sort and remove duplicate timestamps before nearest-neighbour matching.
all_ego_stamp = ego_pose.stamp(:);
[ego_stamp,unique_idx] = unique(all_ego_stamp(valid_ego));
valid_ego = valid_ego(unique_idx);
if isscalar(ego_stamp)
    matched = repmat(valid_ego,numel(messages),1);
else
    matched = interp1(ego_stamp,double(valid_ego),det_stamp(messages),'nearest','extrap');
end
matched = matched(:);
pose_dt = abs(all_ego_stamp(matched)-det_stamp(messages));
lid_det_map.pose_dt(messages) = pose_dt;
accepted = pose_dt <= max_pose_dt;
messages = messages(accepted);
matched = matched(accepted);
lid_det_map.ego_idx(messages) = matched;
if isempty(messages)
    return;
end

%  Static lidar-to-ego transform

ego_frame = urdf_tree.RootFrame;
if isfield(ego_pose,'child_frame_id')
    ego_frame = ego_pose.child_frame_id;
end
if strlength(string(ego_frame)) == 0
    error('load_perception_map:MissingEgoFrame', ...
        'Supply ego_pose.child_frame_id when the static tree has multiple roots.');
end
T = urdf_tree.getTransform(lid_det.frame_id_rel,ego_frame);

%  Convert each message with its matched ego pose

for i = 1:numel(messages)
    m = messages(i);
    k = matched(i);
    R_ego = rpy_rotation(ego_pose.roll(k),ego_pose.pitch(k),ego_pose.yaw(k));
    R = R_ego*T(1:3,1:3);
    position = R_ego*T(1:3,4) + [ego_pose.x(k);ego_pose.y(k);ego_pose.z(k)];

    centers = R*[lid_det.x_cog(m,:);lid_det.y_cog(m,:);lid_det.z_cog(m,:)] + position;
    valid = isfinite(lid_det.x_cog(m,:)) & isfinite(lid_det.y_cog(m,:)) & isfinite(lid_det.z_cog(m,:));
    centers(:,~valid) = NaN;
    lid_det_map.x_map(m,:)   = centers(1,:);
    lid_det_map.y_map(m,:)   = centers(2,:);
    lid_det_map.z_map(m,:)   = centers(3,:);

    % Object rotation is R * Rz(relative yaw); extract its full map RPY.
    c = cosd(lid_det.orientation(m,:));
    s = sind(lid_det.orientation(m,:));
    r11 = R(1,1)*c + R(1,2)*s;
    r21 = R(2,1)*c + R(2,2)*s;
    r31 = R(3,1)*c + R(3,2)*s;
    r32 = -R(3,1)*s + R(3,2)*c;
    cp = hypot(r11,r21);
    roll = atan2d(r32,R(3,3));
    pitch = atan2d(-r31,cp);
    yaw = atan2d(r21,r11);
    singular = cp <= 1e-12;
    roll(singular) = 0;
    yaw(singular) = atan2d(R(1,1)*s(singular)-R(1,2)*c(singular), ...
                         -R(2,1)*s(singular)+R(2,2)*c(singular));
    roll(~valid) = NaN; pitch(~valid) = NaN; yaw(~valid) = NaN;
    lid_det_map.roll_map(m,:) = roll;
    lid_det_map.pitch_map(m,:) = pitch;
    lid_det_map.orientation_map(m,:) = yaw;

    if has_bbox
        x = lid_det.bbox.x(m,:,:);
        y = lid_det.bbox.y(m,:,:);
        z = lid_det.bbox.z(m,:,:);
        vertices = R*[x(:).';y(:).';z(:).'] + position;
        lid_det_map.bbox_map.x(m,:,:) = reshape(vertices(1,:),size(x));
        lid_det_map.bbox_map.y(m,:,:) = reshape(vertices(2,:),size(y));
        lid_det_map.bbox_map.z(m,:,:) = reshape(vertices(3,:),size(z));
    end
end

end % FUNCTION END

function R = rpy_rotation(roll,pitch,yaw)
c = cosd([roll pitch yaw]);
s = sind([roll pitch yaw]);
R = [c(3)*c(2),c(3)*s(2)*s(1)-s(3)*c(1),c(3)*s(2)*c(1)+s(3)*s(1); ...
     s(3)*c(2),s(3)*s(2)*s(1)+c(3)*c(1),s(3)*s(2)*c(1)-c(3)*s(1); ...
     -s(2),c(2)*s(1),c(2)*c(1)];
end
