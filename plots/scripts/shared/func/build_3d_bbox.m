function det_struct = build_3d_bbox(det_struct)
% BUILD_3D_BBOX  Build oriented 3D bounding boxes for detections.
%
%   The resulting bounding boxes are stored in:
%       det_struct.bbox.x
%       det_struct.bbox.y
%       det_struct.bbox.z
%
%   Dimensions: [N_messages, N_detections, N_vertices]
%
%   Assumptions:
%       - Position refers to the geometric centre of the box.
%       - Orientation is the yaw angle in degrees.
%       - Length is aligned with the local X axis.
%       - Width is aligned with the local Y axis.
%       - Height is aligned with the local Z axis.

%  Define bounding box vertices

% Local corners (normalised to half-dimensions)
corners = [
     1   1  -1;
    -1   1  -1;
    -1  -1  -1;
     1  -1  -1;
     1   1   1;
    -1   1   1;
    -1  -1   1;
     1  -1   1
];

% Vertex sequence for wireframe plotting
% NaN separators are introduced using zero indices
path = [
    1 2 3 4 1 0 ...      % Bottom face
    5 6 7 8 5 0 ...      % Top face
    1 5 0 ...            % Vertical edge 1
    2 6 0 ...            % Vertical edge 2
    3 7 0 ...            % Vertical edge 3
    4 8 0               % Vertical edge 4
];

% Build normalised coordinate sequences
vertices = nan(numel(path), 3);
valid_path = path > 0;
vertices(valid_path, :) = corners(path(valid_path), :);

sx = reshape(vertices(:, 1), 1, 1, []);
sy = reshape(vertices(:, 2), 1, 1, []);
sz = reshape(vertices(:, 3), 1, 1, []);

%  Extract detection information

x = det_struct.x_cog;
y = det_struct.y_cog;
z = det_struct.z_cog;

L = det_struct.length / 2;
W = det_struct.width  / 2;
H = det_struct.height / 2;

yaw = det_struct.orientation;

% Invalid detections
valid = isfinite(x) & isfinite(y) & isfinite(z) & ...
        isfinite(L) & isfinite(W) & isfinite(H) & ...
        isfinite(yaw) & L > 0 & W > 0 & H > 0;

x(~valid) = NaN;
y(~valid) = NaN;
z(~valid) = NaN;

%  Build oriented bounding boxes

c = cosd(yaw);
s = sind(yaw);

det_struct.bbox.x = x + L .* c .* sx - W .* s .* sy;
det_struct.bbox.y = y + L .* s .* sx + W .* c .* sy;
det_struct.bbox.z = z + H .* sz;

end % FUNCTION END