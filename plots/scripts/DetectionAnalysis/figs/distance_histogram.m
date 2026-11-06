% Detection distance from the ego vehicle, using the matched map-frame pose.
% Counts and per-range-bin medians refer to the current filtered detections.
distance_figure = figure("Name","DISTANCE",'NumberTitle','off');
tiledlayout(3,1,'Padding','compact');

class_names = fieldnames(DET_LABELS);
class_ids = cellfun(@(name) DET_LABELS.(name),class_names);
class_names = class_names(ismember(class_ids,VIS_CLASSES));
bin_width = 5; % Distance bin width [m].
max_distance = 0;
sensor_distances = cell(size(sensors));
sensor_valid = cell(size(sensors));

% Distance from the matched ego pose
for i = 1:numel(sensors)
    s = sensors{i}.s;

    ego_x = nan(size(s.class,1),1);
    ego_y = nan(size(s.class,1),1);
    valid_pose = isfinite(s.ego_idx);
    ego_x(valid_pose) = ego.x(s.ego_idx(valid_pose));
    ego_y(valid_pose) = ego.y(s.ego_idx(valid_pose));
    distance = hypot(s.x_map - ego_x,s.y_map - ego_y);

    valid = ...
        isfinite(s.stamp) & ...
        isfinite(distance) & ...
        ismember(s.class,VIS_CLASSES);

    sensor_distances{i} = distance;
    sensor_valid{i} = valid;
    valid_distance = distance(valid);
    max_distance = max([max_distance; valid_distance(:)]);
end

% Common bins for all sensors, selected classes and panels.
bin_edges = 0:bin_width:max(bin_width,ceil(max_distance/bin_width)*bin_width);
n_bins = numel(bin_edges) - 1;

% Count
ax_distance_count = nexttile; hold on;
for i = 1:numel(sensors)
    s = sensors{i}.s;
    distance = sensor_distances{i};
    valid = sensor_valid{i};

    for j = 1:numel(class_names)
        class_name = class_names{j};
        class_idx = valid & s.class == DET_LABELS.(class_name);
        if ~any(class_idx(:))
            continue;
        end

        histogram( ...
            distance(class_idx), ...
            'BinEdges', bin_edges, ...
            'Normalization', 'count', ...
            'DisplayStyle', 'stairs', ...
            'LineWidth', 1.5, ...
            'EdgeColor', col.det.(class_name), ...
            'DisplayName', sprintf('%s - %s', ...
                sensors{i}.name, class_name));
    end
end
xlabel("detection range [m]")
ylabel("count [-]")
grid on;
legend show

% Median confidence
ax_confidence_range = nexttile; hold on;
for i = 1:numel(sensors)
    s = sensors{i}.s;
    distance = sensor_distances{i};
    valid = sensor_valid{i} & isfinite(s.score);

    for j = 1:numel(class_names)
        class_name = class_names{j};
        class_idx = valid & s.class == DET_LABELS.(class_name);
        if ~any(class_idx(:))
            continue;
        end

        [~, ~, range_bins] = histcounts(distance(class_idx),bin_edges);
        values = double(s.score(class_idx));
        median_confidence = accumarray(range_bins(:),values(:),[n_bins 1],@mean,NaN);

        % Extend the final bin to its right edge; empty bins remain gaps.
        stairs( ...
            bin_edges, [median_confidence; median_confidence(end)], ...
            'LineWidth', 1.5, ...
            'Color', col.det.(class_name), ...
            'DisplayName', sprintf('%s - %s', ...
                sensors{i}.name, class_name));
    end
end
xlabel("detection range [m]")
ylabel("median conf [-]")
ylim([0 1]);
grid on;
legend show

% Median point count
ax_point_count_range = nexttile; hold on;
for i = 1:numel(sensors)
    s = sensors{i}.s;
    distance = sensor_distances{i};
    valid = sensor_valid{i} & isfinite(s.pt_count);

    for j = 1:numel(class_names)
        class_name = class_names{j};
        class_idx = valid & s.class == DET_LABELS.(class_name);
        if ~any(class_idx(:))
            continue;
        end

        [~, ~, range_bins] = histcounts(distance(class_idx),bin_edges);
        values = double(s.pt_count(class_idx));
        median_points = accumarray(range_bins(:),values(:),[n_bins 1],@mean,NaN);

        stairs( ...
            bin_edges, [median_points; median_points(end)], ...
            'LineWidth', 1.5, ...
            'Color', col.det.(class_name), ...
            'DisplayName', sprintf('%s - %s', ...
                sensors{i}.name, class_name));
    end
end
xlabel("detection range [m]")
ylabel("median pt_count [-]")
grid on;
legend show

linkaxes([ax_distance_count,ax_confidence_range,ax_point_count_range],'x');
xlim(ax_distance_count,[bin_edges(1),bin_edges(end)]);
