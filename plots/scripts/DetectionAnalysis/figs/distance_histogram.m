% Detection distance from the ego vehicle, using the matched map-frame pose.
% Counts and confidence distributions use the current filtered detections.
distance_figure = figure("Name","DISTANCE",'NumberTitle','off');
tiledlayout(2,1,'Padding','compact');

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

% Grouped confidence violins, medians and 25th-75th percentile whiskers.
ax_confidence_range = nexttile; hold on;
bin_centers = (bin_edges(1:end-1) + bin_edges(2:end))/2;
n_series = numel(sensors)*numel(class_names);
series_width = 0.8*bin_width/max(1,n_series);
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
        median_confidence = accumarray(range_bins(:),values(:),[n_bins 1],@median,NaN);
        lower_confidence = accumarray(range_bins(:),values(:),[n_bins 1], ...
            @(v) prctile(v,25),NaN);
        upper_confidence = accumarray(range_bins(:),values(:),[n_bins 1], ...
            @(v) prctile(v,75),NaN);

        % Reserve a side-by-side slot for each sensor/class in every bin.
        series_idx = (i - 1)*numel(class_names) + j;
        offset = (series_idx - (n_series + 1)/2)*series_width;
        populated_bins = find(isfinite(median_confidence));
        violin_centers = reshape(bin_centers(populated_bins),[],1) + offset;
        half_width = 0.45*series_width;
        medians = median_confidence(populated_bins);
        lower = lower_confidence(populated_bins);
        upper = upper_confidence(populated_bins);
        % One legend entry per sensor/class, including constant distributions.
        patch(ax_confidence_range, ...
            'XData', nan(4,1), 'YData', nan(4,1), ...
            'FaceColor', col.det.(class_name), ...
            'EdgeColor', 'k', ...
            'LineWidth', 1, ...
            'DisplayName', sprintf('%s - %s', ...
                sensors{i}.name, class_name));

        for k = 1:numel(populated_bins)
            bin_values = values(range_bins == populated_bins(k));
            center = violin_centers(k);
            if min(bin_values) == max(bin_values)
                % A singleton or constant sample has no density to estimate.
                plot(ax_confidence_range, ...
                    center + [-half_width half_width], ...
                    [bin_values(1) bin_values(1)], ...
                    'Color', col.det.(class_name), ...
                    'LineWidth', 3, 'HandleVisibility', 'off');
                continue;
            end

            confidence_grid = linspace(min(bin_values),max(bin_values),100);
            density = ksdensity(bin_values,confidence_grid);
            width = half_width*density/max(density);
            patch(ax_confidence_range, ...
                'XData', center + [-width fliplr(width)], ...
                'YData', [confidence_grid fliplr(confidence_grid)], ...
                'FaceColor', col.det.(class_name), ...
                'EdgeColor', 'k', ...
                'LineWidth', 1, 'HandleVisibility', 'off');
        end

        % Overlay percentile whiskers and median markers on the violins.
        errorbar(ax_confidence_range, ...
            violin_centers, medians, medians - lower, upper - medians, ...
            'LineStyle', 'none', ...
            'Color', 'k', ...
            'LineWidth', 1.2, ...
            'CapSize', 6, ...
            'HandleVisibility', 'off');

        plot(ax_confidence_range, ...
            violin_centers, medians, 'ko', ...
            'MarkerFaceColor', 'w', 'MarkerSize', 4, ...
            'HandleVisibility', 'off');
    end
end

% Shared summary keys apply to every class; NaNs keep them off the plot.
plot(ax_confidence_range, NaN, NaN, 'ko', ...
    'MarkerFaceColor', 'w', 'MarkerSize', 4, 'DisplayName', 'Median');
errorbar(ax_confidence_range, NaN, NaN, NaN, NaN, ...
    'LineStyle', '-', ...
    'Color', 'k', ...
    'LineWidth', 1.2, ...
    'CapSize', 6, ...
    'DisplayName', '25th-75th percentiles');
xlabel("detection range [m]")
ylabel("confidence [-]")
ylim([0 1]);
grid on;
legend show

linkaxes([ax_distance_count,ax_confidence_range],'x');
xlim(ax_distance_count,[bin_edges(1),bin_edges(end)]);
